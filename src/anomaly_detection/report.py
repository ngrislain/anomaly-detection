"""Render a standalone HTML page comparing character-level anomaly scores.

Each character is shaded by its log-probability against a pair of bounds: the
character at the upper bound is left fully transparent and one at the lower
bound is solid fuchsia.

Those bounds slide, via `beta`, between one range shared by the whole report
(the n-gram's range on the unaffected text, making every panel directly
comparable) and the range observed on each panel's own text (full contrast, but
panels no longer comparable). Each bound can then be stretched by a scale
factor, and the interpolated surprise raised to the power `alpha`, which pulls
the mid-range towards transparent and concentrates colour on the anomalies.

Separately, and independently of any model, text that was spliced into the
evaluated text sits on a pale wash marking where it really is, so a reader can
tell a model's verdict from the ground truth.
"""

import html
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from string import Template

from anomaly_detection.detector import detect_anomalies
from anomaly_detection.models import CharLogProbModel, EwmaModel, NgramModel, QwenModel

# Factors stretching the colour-scale bounds: log-probabilities are never
# positive, so a factor above 1 pushes a bound further from zero and widens the
# range, softening the shading.
DEFAULT_MIN_SCALE = 1.0
DEFAULT_MAX_SCALE = 1.0

# How far each panel scales to its own text rather than to the unaffected
# reference text: 0 gives the whole report one shared scale, 1 gives each panel
# its own. Defaults to 1, which keeps each panel's full contrast.
DEFAULT_BETA = 1.0

# Exponent applied to the interpolated surprise. 1.0 shades linearly between
# the bounds; higher values pull the mid-range towards transparent so only the
# genuinely surprising characters stay coloured.
DEFAULT_ALPHA = 1.0

# Default order of the n-gram model.
DEFAULT_N = 5

# How many of the fitted model's commonest grams to tabulate; 0 omits the table.
DEFAULT_TOP_NGRAMS = 20

# Half-life, in characters, of the EWMA smoothing applied to n-gram scores.
# Neural scores are already smooth across a token, so they are left alone.
DEFAULT_HALF_LIFE = 2.0

# Number of quantised shading steps. Adjacent characters sharing an identical
# log-probability collapse into one <span>, which keeps the generated page
# small -- especially for Qwen, where every character of a token scores alike.
_LEVELS = 24

# The shading is a single fuchsia whose opacity tracks surprise: characters at
# the upper bound are fully transparent, those at the lower bound solid fuchsia.
# Held back from the blue corner so it reads pink-magenta rather than violet.
_HIGHLIGHT = (255, 80, 120)
_MAX_ALPHA = 1.0

# Flat wash marking where text was actually inserted, independent of what any
# model thinks. Pale enough that the fuchsia shading composites cleanly on top.
_INSERTED_WASH = "#fdf6e9"


def _ramp_colour(t: float) -> tuple[int, int, int, float]:
    """Highlight colour at `t` in [0, 1], where 1 is the most surprising."""
    return (*_HIGHLIGHT, _MAX_ALPHA * t)


def _levels_css() -> str:
    rules = []
    for level in range(1, _LEVELS):
        r, g, b, alpha = _ramp_colour(level / (_LEVELS - 1))
        rules.append(f".l{level}{{background:rgba({r},{g},{b},{alpha:.3f})}}")
    return "\n".join(rules)


def _legend_gradient() -> str:
    stops = []
    for level in range(_LEVELS):
        r, g, b, alpha = _ramp_colour(level / (_LEVELS - 1))
        # Reversed: the legend reads min_log_prob (surprising) on the left.
        stops.append(f"rgba({r},{g},{b},{alpha:.3f}) {100 * (1 - level / (_LEVELS - 1)):.1f}%")
    return "linear-gradient(90deg, " + ", ".join(reversed(stops)) + ")"


def _shade_level(log_prob: float, min_log_prob: float, max_log_prob: float, alpha: float) -> int:
    """Map a log-probability to a shading step (0 = no highlight)."""
    if not math.isfinite(log_prob):
        return 0
    span = max_log_prob - min_log_prob
    t = 1.0 if span <= 0 else (log_prob - min_log_prob) / span
    surprise = 1.0 - min(1.0, max(0.0, t))
    return round((surprise**alpha) * (_LEVELS - 1))


def _render_scored(
    pairs: list[tuple[str, float]],
    min_log_prob: float,
    max_log_prob: float,
    alpha: float,
    inserted: tuple[int, int] | None = None,
) -> str:
    """Turn scored characters into HTML.

    Only characters carrying an identical log-probability are merged into a
    single span, so the value revealed on hover is always that character's own
    score rather than an average or an extreme of its neighbours.

    `inserted` is a character range wrapped in a flat wash marking text that was
    spliced in. Runs are broken at its edges so the wash nests cleanly, and the
    shading spans inside it composite over the wash rather than replacing it.
    """
    out: list[str] = []
    run_chars: list[str] = []
    run_log_prob: float | None = None

    def flush() -> None:
        if not run_chars:
            return
        text = html.escape("".join(run_chars))
        if run_log_prob is None or not math.isfinite(run_log_prob):
            out.append(text)
            return
        level = _shade_level(run_log_prob, min_log_prob, max_log_prob, alpha)
        css = f' class="l{level}"' if level else ""
        out.append(f'<span{css} data-lp="{run_log_prob:.3f}">{text}</span>')

    for index, (char, log_prob) in enumerate(pairs):
        boundary = inserted is not None and index in inserted
        same = (
            not boundary
            and run_log_prob is not None
            and (log_prob == run_log_prob or (math.isnan(log_prob) and math.isnan(run_log_prob)))
        )
        if not same:
            flush()
            run_chars, run_log_prob = [], log_prob
        if boundary:
            out.append('<mark class="ins">' if index == inserted[0] else "</mark>")
        run_chars.append(char)
    flush()
    # An insertion running to the very end never hits its closing index above.
    if inserted is not None and inserted[1] >= len(pairs) > inserted[0]:
        out.append("</mark>")
    return "".join(out)


def _alternatives(
    alt_text: str | Sequence[str] | Mapping[str, str] | None,
) -> dict[str, str]:
    """Normalise the `alt_text` argument into an ordered heading -> text map."""
    if alt_text is None:
        return {}
    if isinstance(alt_text, str):
        return {"Alternative text": alt_text}
    if isinstance(alt_text, Mapping):
        return dict(alt_text)
    return {f"Alternative text {i}": text for i, text in enumerate(alt_text, start=1)}


# One rendered panel: title, scores, bounds, hover label, inserted-text span.
_PanelSpec = tuple[
    str, list[tuple[str, float]], float, float, str, tuple[int, int] | None
]


def _inserted_span(base: str, variant: str) -> tuple[int, int] | None:
    """Character range of what `variant` adds to `base`, or None.

    Derived by matching the common prefix and suffix, which pins the insertion
    exactly when `variant` is `base` with one contiguous run added -- the case
    the report is built around. Returns None for anything else, so an unrelated
    text is simply left unmarked rather than mis-marked.
    """
    if len(variant) <= len(base):
        return None
    prefix = 0
    while prefix < len(base) and base[prefix] == variant[prefix]:
        prefix += 1
    suffix = 0
    while suffix < len(base) - prefix and base[-1 - suffix] == variant[-1 - suffix]:
        suffix += 1
    if prefix + suffix != len(base):
        return None
    return (prefix, len(variant) - suffix)


def _visible(gram: str) -> str:
    """Escape a gram and make its whitespace legible in a table cell."""
    return html.escape(gram).replace("\n", "⏎").replace("\t", "⇥").replace(" ", "·")


def _ngram_table(model: NgramModel, k: int) -> str:
    """Table of the model's most frequent grams, or "" if none are wanted."""
    rows = model.top_ngrams(k)
    if not rows:
        return ""
    body = "".join(
        f"<tr><td class=num>{rank}</td>"
        f'<td><code>{_visible(gram)}</code></td>'
        f"<td class=num>{count:,}</td>"
        f"<td class=num>{100 * frequency:.3f}%</td></tr>"
        for rank, (gram, count, frequency) in enumerate(rows, start=1)
    )
    return (
        "<section class=wrap>"
        f"<h2>Most frequent {model.n}-grams</h2>"
        f'<p class="note">The {len(rows)} commonest grams in the fitting text, with their '
        "share of all scored positions. Spaces are shown as ·, newlines as ⏎.</p>"
        "<table><thead><tr><th class=num>#</th><th>gram</th>"
        "<th class=num>count</th><th class=num>frequency</th></tr></thead>"
        f"<tbody>{body}</tbody></table></section>"
    )


def _extra_targets(
    extra_on: str | Sequence[str] | None, available: dict[str, str]
) -> dict[str, str]:
    """Pick which texts the extra models score, defaulting to all of them."""
    if extra_on is None:
        return available
    wanted = [extra_on] if isinstance(extra_on, str) else list(extra_on)
    unknown = [h for h in wanted if h not in available]
    if unknown:
        raise ValueError(
            f"extra_on refers to unknown section(s) {unknown}; "
            f"available: {list(available)}"
        )
    return {heading: available[heading] for heading in wanted}


def _mean_log_prob(pairs: list[tuple[str, float]]) -> float:
    values = [v for _, v in pairs if math.isfinite(v)]
    return sum(values) / len(values) if values else math.nan


def _observed_bounds(pairs: list[tuple[str, float]], warmup: int = 0) -> tuple[float, float]:
    """Lowest and highest finite score in `pairs`, before any scaling.

    The first `warmup` characters are skipped. An order-n model scores those
    positions against sentinel padding rather than real text, so they are
    reliably the most surprising thing in any document and would otherwise peg
    the lower bound to an artifact of where scoring starts. They are still
    displayed and shaded -- they just do not get to define the scale.
    """
    considered = pairs[warmup:] or pairs
    values = [v for _, v in considered if math.isfinite(v)]
    if not values:
        return (-1.0, 0.0)
    return (min(values), max(values))


def _blend_bounds(
    reference: tuple[float, float],
    own: tuple[float, float],
    beta: float,
    min_scale: float,
    max_scale: float,
) -> tuple[float, float]:
    """Mix reference and own bounds, then stretch them by the scale factors.

    `beta` slides continuously from 0, where every panel shares one reference
    range, to 1, where each panel is scaled to its own text. The shared end
    makes shading comparable across the whole report; the self end gives each
    panel its full dynamic range.

    Log-probabilities are never positive, so a scale factor above 1 moves a
    bound further from zero, widening the range and softening the shading.
    """
    low = (reference[0] + (own[0] - reference[0]) * beta) * min_scale
    high = (reference[1] + (own[1] - reference[1]) * beta) * max_scale
    # Scaling the two ends independently can invert them, which would leave the
    # panel uniformly transparent; keep a non-degenerate range instead.
    return (min(low, high - 1e-9), high)


def _panel(
    title: str,
    pairs: list[tuple[str, float]],
    min_lp: float,
    max_lp: float,
    alpha: float,
    label: str = "log p",
    inserted: tuple[int, int] | None = None,
) -> str:
    mean = _mean_log_prob(pairs)
    mean_text = "n/a" if math.isnan(mean) else f"{mean:.3f}"
    return (
        '<figure class="panel">'
        f'<figcaption><span class="panel-name">{html.escape(title)}</span>'
        f'<span class="stat">mean {mean_text} · range {min_lp:.2f}…{max_lp:.2f}</span>'
        "</figcaption>"
        f'<div class="box scored" data-label="{html.escape(label)}">'
        f"{_render_scored(pairs, min_lp, max_lp, alpha, inserted)}</div>"
        "</figure>"
    )


def _section(
    heading: str,
    note: str,
    panels_spec: Sequence[_PanelSpec],
    alpha: float,
) -> str:
    """Render one section from already-scored panels.

    Each panel carries its own title, scores, bounds and label, so a section can
    show many models over one text or one model over many texts.
    """
    panels = "".join(
        _panel(title, pairs, min_lp, max_lp, alpha, label, inserted)
        for title, pairs, min_lp, max_lp, label, inserted in panels_spec
    )
    return (
        "<section>"
        f'<div class="wrap"><h2>{html.escape(heading)}</h2>'
        f'<p class="note">{html.escape(note)}</p></div>'
        f'<div class="breakout grid">{panels}</div>'
        "</section>"
    )


_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
:root{
  --bg:#ffffff;
  --panel:#ffffff;
  --ink:#33383d;
  --muted:#6b7278;
  --rule:#e4e7ea;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
}
*{box-sizing:border-box}
body{
  margin:0;
  padding:3.5rem 1.25rem 5rem;
  background:var(--bg);
  color:var(--ink);
  font-family:var(--sans);
  font-size:16px;
  line-height:1.65;
}
/* Only the horizontal margins, so this never clobbers section spacing. */
.wrap{max-width:44rem;margin-left:auto;margin-right:auto}
/* Comparison panels break out past the prose column on both sides. */
.breakout{width:min(78rem,96vw);margin-left:50%;transform:translateX(-50%)}
/* Serif is reserved for headings; everything else is sans. */
h1,h2{font-family:var(--serif)}
h1{font-size:2rem;font-weight:600;letter-spacing:-.01em;margin:0 0 .4rem}
h2{font-size:1.3rem;font-weight:600;margin:0 0 .3rem}
section{margin-top:3rem}
.sub,.note{color:var(--muted);margin:0 0 1rem;font-size:.9rem}
.box{
  background:var(--panel);
  border-radius:8px;
  padding:1.1rem 1.25rem;
  box-shadow:0 1px 2px rgba(16,24,40,.06), 0 4px 12px rgba(16,24,40,.07);
  overflow:auto;
  white-space:pre-wrap;
  overflow-wrap:break-word;
}
.fit{max-height:15rem;color:var(--muted);font-size:.9rem}
/* Constant height so the two models stay visually comparable. */
.scored{height:26rem}
.grid{display:grid;gap:1.25rem;grid-template-columns:repeat(auto-fit,minmax(20rem,1fr))}
.panel{margin:0;min-width:0}
figcaption{
  display:flex;justify-content:space-between;align-items:baseline;
  gap:1rem;margin-bottom:.45rem;
}
.panel-name{font-weight:600}
.stat{color:var(--muted);font-size:.85rem;font-variant-numeric:tabular-nums}
.legend{margin:1.75rem 0 0;padding-top:1.25rem;border-top:1px solid var(--rule)}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:.34rem .6rem;border-bottom:1px solid var(--rule);text-align:left}
th{font-weight:600;color:var(--muted);font-size:.8rem;letter-spacing:.02em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;width:1%;white-space:nowrap}
/* Grams are exact character sequences, so they need a fixed pitch. */
table code{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.88rem;white-space:pre;
}
.scored span{cursor:help}
/* Ground truth: where text was actually spliced in. A flat wash under the
   shading, so the two readings stay distinguishable. */
mark.ins{background:$wash;color:inherit}
/* Instant hover readout: the native title tooltip is too slow for scanning. */
#tip{
  position:fixed;z-index:10;pointer-events:none;opacity:0;
  transition:opacity .08s ease-out;
  background:var(--ink);color:var(--bg);
  padding:.15rem .45rem;border-radius:4px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.75rem;font-variant-numeric:tabular-nums;
}
.bar{height:.6rem;border-radius:3px;background:$gradient;border:1px solid var(--rule)}
.scale{
  display:flex;justify-content:space-between;
  color:var(--muted);font-size:.82rem;margin-top:.35rem;
  font-variant-numeric:tabular-nums;
}
$levels
</style>
</head>
<body>
<main>
<header class="wrap">
  <h1>$title</h1>
  <p class="sub">Each character is shaded by its log-probability given every
  character before it, against the range each panel states beside its name:
  characters at the top of that range stay transparent, those at the bottom
  turn solid fuchsia. The interpolated surprise is raised to the power $alpha,
  which pulls the mid-range towards transparent so colour concentrates on the
  anomalies. $scope Where text was actually spliced in, it sits on a pale
  yellow wash — that is ground truth, not a model judgement, so the fuchsia
  should ideally land on top of it. Hover any character to read its value.</p>
  <div class="legend">
    <div class="bar"></div>
    <div class="scale">
      <span>lowest in panel</span><span>highest in panel</span>
    </div>
  </div>
</header>
<section class="wrap">
  <h2>$ngram_name fitting text</h2>
  <p class="note">$fit_chars characters used to fit the $ngram_name model.</p>
  <div class="box fit">$fit_text</div>
</section>
$ngram_table
$sections
</main>
<div id="tip"></div>
<script>
(function(){
  var tip = document.getElementById('tip');
  document.addEventListener('mousemove', function(e){
    var el = e.target.closest ? e.target.closest('[data-lp]') : null;
    if (!el) { tip.style.opacity = '0'; return; }
    var box = el.closest('.scored');
    tip.textContent = ((box && box.dataset.label) || 'log p') + ' = ' + el.dataset.lp;
    tip.style.opacity = '1';
    // Flip to the left of the cursor near the right edge so it stays on screen.
    var w = tip.offsetWidth;
    var x = e.clientX + 14;
    if (x + w > window.innerWidth - 8) x = e.clientX - w - 14;
    tip.style.left = x + 'px';
    tip.style.top = (e.clientY + 16) + 'px';
  });
})();
</script>
</body>
</html>
"""
)


def build_report(
    fit_text: str,
    eval_text: str,
    min_scale: float = DEFAULT_MIN_SCALE,
    max_scale: float = DEFAULT_MAX_SCALE,
    beta: float = DEFAULT_BETA,
    alpha: float = DEFAULT_ALPHA,
    alt_text: str | Sequence[str] | Mapping[str, str] | None = None,
    out_dir: str | Path = "out",
    filename: str = "anomaly_report.html",
    title: str = "Character-level anomaly report",
    n: int = DEFAULT_N,
    top_ngrams: int = DEFAULT_TOP_NGRAMS,
    half_life: float | None = DEFAULT_HALF_LIFE,
    ngram: NgramModel | None = None,
    qwen: CharLogProbModel | None = None,
    extra_models: Mapping[str, CharLogProbModel] | None = None,
    extra_on: str | Sequence[str] | None = None,
) -> Path:
    """Write a standalone HTML comparison of n-gram and Qwen scores.

    Args:
        fit_text: text the n-gram model is fitted on.
        eval_text: text to score and display.
        min_scale: factor applied to each panel's lowest observed
            log-probability to set the point where shading saturates. Above 1
            widens the range and softens the shading; below 1 tightens it.
        max_scale: same, for each panel's highest observed log-probability,
            the point at or above which nothing is shaded.
        beta: slides the colour-scale bounds between two regimes, continuously.
            At 0 every panel in the report -- both models, every text -- shares
            one range, taken from the n-gram on `eval_text`, the unaffected
            text, so all panels are directly comparable. At 1 each panel is
            scaled to its own text, giving it full contrast but making panels
            incomparable.
        alpha: exponent applied to the interpolated surprise. Values above 1
            concentrate the shading on the most surprising characters; 1 shades
            linearly between the bounds.
        alt_text: optional further text(s) to score, typically `eval_text` with
            an anomaly inserted, each shown in its own section. Accepts a
            single string, a sequence of strings, or a mapping of section
            heading to text.
        out_dir: directory to write into; created if missing.
        filename: name of the generated page.
        title: page heading.
        n: order of the n-gram model, ignored when `ngram` is supplied.
        top_ngrams: how many of the fitted model's commonest grams to tabulate
            below the fitting text. 0 omits the table.
        half_life: half-life in characters of the causal EWMA smoothing applied
            to n-gram scores only, turning spiky per-character values into
            sustained regions. Neural models are left unsmoothed. Pass None to
            score every model's raw log-probabilities.
        ngram: pre-fitted n-gram model; one of order `n` is fitted on
            `fit_text` when omitted.
        qwen: pre-loaded Qwen model, useful to avoid reloading weights.
        extra_models: further named models to show as additional panels in
            every section, e.g. an n-gram fitted on a different language so the
            contrast between the two training distributions is visible. They
            are smoothed with the same `half_life` as the built-in models.
        extra_on: which section heading(s) the extra models should score. A
            single heading keeps the closing section to one panel. Defaults to
            every alternative text.

    Returns:
        Path to the written HTML file.
    """
    if min_scale <= 0 or max_scale <= 0:
        raise ValueError(f"scales must be positive, got {min_scale} and {max_scale}")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be between 0 and 1, got {beta}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    if ngram is None:
        ngram = NgramModel(n=n).fit(fit_text)
    if qwen is None:
        qwen = QwenModel()

    def prepare(name: str, model: CharLogProbModel) -> tuple[str, CharLogProbModel, str]:
        """Smooth n-gram scores only; neural scores are already smooth across a
        token, and smoothing them would blur the token boundaries away.

        Smoothed values average log-probabilities, so they are no longer log p
        and are labelled as scores.
        """
        if half_life is not None and isinstance(model, NgramModel):
            return (name, EwmaModel(model, half_life=half_life), "score")
        return (name, model, "log p")

    ngram_name = f"{ngram.n}-gram"
    head_to_head = [
        prepare(ngram_name, ngram),
        prepare(getattr(qwen, "model_name", type(qwen).__name__), qwen),
    ]
    extra = [prepare(name, m) for name, m in (extra_models or {}).items()]

    alternatives = _alternatives(alt_text)
    texts = {"Evaluated text": eval_text, **alternatives}

    scored: dict[tuple[int, str], list[tuple[str, float]]] = {}

    def score(model: CharLogProbModel, text: str) -> list[tuple[str, float]]:
        """Score once per (model, text); `eval_text` is reused as the reference."""
        key = (id(model), text)
        if key not in scored:
            scored[key] = detect_anomalies(model, text)
        return scored[key]

    # Positions the n-gram scores against padding rather than real text. Kept
    # out of every bound, reference and own alike, so neither end of the beta
    # range is pegged to that artifact.
    warmup = max(ngram.n - 1, 0)

    # One reference for the whole report: the n-gram's range on the unaffected
    # text. At beta = 0 every panel uses it, so all of them -- n-gram and neural
    # alike, across every text -- share a single scale and are directly
    # comparable. At beta = 1 each panel falls back to its own range.
    reference = _observed_bounds(score(head_to_head[0][1], eval_text), warmup)

    def spec(title: str, text: str, model: CharLogProbModel, label: str) -> _PanelSpec:
        pairs = score(model, text)
        min_lp, max_lp = _blend_bounds(
            reference, _observed_bounds(pairs, warmup), beta, min_scale, max_scale
        )
        # Ground truth, recovered by diffing against the unaffected text, so no
        # caller-supplied offset can drift out of step with the text.
        return (title, pairs, min_lp, max_lp, label, _inserted_span(eval_text, text))

    # One head-to-head section per text, every model over the same text.
    sections = [
        _section(
            heading,
            f"{len(text):,} characters, scored by each model.",
            [spec(name, text, model, label) for name, model, label in head_to_head],
            alpha,
        )
        for heading, text in texts.items()
    ]

    # Extra models close the report rather than crowding every head-to-head.
    if extra:
        targets = _extra_targets(extra_on, alternatives or texts)
        multiple = len(targets) > 1
        shown = "The texts above" if multiple else f"“{next(iter(targets))}”"
        sections.append(
            _section(
                "Fitted on another corpus",
                f"{shown}, scored by a model fitted on a different corpus.",
                [
                    spec(f"{name} · {heading}" if multiple else name, text, model, label)
                    for name, model, label in extra
                    for heading, text in targets.items()
                ],
                alpha,
            )
        )

    page = _TEMPLATE.substitute(
        title=html.escape(title),
        gradient=_legend_gradient(),
        levels=_levels_css(),
        wash=_INSERTED_WASH,
        alpha=f"{alpha:g}",
        scope=(
            "Every panel shares one range, taken from the n-gram on the "
            "unaffected text, so all of them are directly comparable."
            if beta == 0.0
            else "Each panel is scaled to its own text, so shading compares "
            "characters within a panel, not across panels."
            if beta == 1.0
            else f"Ranges are blended {1 - beta:g} toward the unaffected text's "
            f"range and {beta:g} toward each panel's own."
        ),
        ngram_name=ngram_name,
        fit_chars=f"{len(fit_text):,}",
        fit_text=html.escape(fit_text),
        ngram_table=_ngram_table(ngram, top_ngrams) if top_ngrams > 0 else "",
        sections="\n".join(sections),
    )

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(page, encoding="utf-8")
    return path
