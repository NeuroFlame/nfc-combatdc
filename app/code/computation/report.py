"""Build each site's self-contained HTML results page."""

import math
from dataclasses import dataclass
from html import escape
from typing import Any, Dict, List, Optional, Sequence

from .diagnostics import (
    between_site_spread,
    collect_findings,
    harmonized_data_is_reliable,
    site_name,
)
from .local_math import roi_mean_sd
from .types import (
    DEFAULT_MIN_SITE_SUMMARY_N,
    DEFAULT_SHARE_SITE_SUMMARIES,
    CrossSiteSummaries,
    SiteState,
)

SITE_COLUMN_PREFIX = "site_"


@dataclass
class _Series:
    name: str
    values: Sequence[float]
    css_class: str


def build_report(
    state: SiteState,
    summaries: CrossSiteSummaries,
    parameters: Dict[str, Any],
    output_file_name: str,
    covariate_file_name: Optional[str] = None,
) -> str:
    """Return the HTML results page for one site.

    Args:
        state: Site state after harmonization.
        summaries: Cross-site summaries (empty when sharing is disabled).
        parameters: Computation parameters.
        output_file_name: Name of the harmonized CSV, for the download link.
            The link is replaced by an explanation when the design matrix is
            rank-deficient, because the CSV is then not written.
        covariate_file_name: Name of this site's copied covariate file, for the
            download link; ``None`` when it was not copied.

    Returns:
        A self-contained HTML document.
    """
    regression = state.regression
    harmonization = state.harmonization
    own_name = site_name(state)
    site_names = [
        column[len(SITE_COLUMN_PREFIX) :] for column in state.layout.site_covar_list
    ]
    # Only sites whose results appear in this report get a pill; other sites'
    # results appear only when the cross-site comparison is shown.
    comparison_shown = summaries.enabled and len(summaries.sites) >= 2
    visible_sites = [
        name
        for name in site_names
        if name == own_name or (comparison_shown and name in summaries.sites)
    ]
    rois = [str(c) for c in state.data.columns]
    covariates_types = parameters.get("covariates_types", {})

    mean_before, sd_before = roi_mean_sd(state.data)
    mean_after, sd_after = roi_mean_sd(harmonization.harmonized)

    header = _header(
        own_name=own_name,
        site_names=visible_sites,
        chips=[
            ("Sites", str(len(site_names))),
            ("Subjects", str(regression.n_sample)),
            ("This site", str(len(state.data))),
            ("ROIs", str(len(rois))),
            ("Covariates", str(len(covariates_types))),
            ("Algorithm", str(parameters.get("combat_algo", "—"))),
        ],
    )
    sections = [
        (
            "diagnostics",
            "Diagnostics and recommendations",
            _diagnostics(state, summaries, parameters),
        ),
        (
            "site-effects",
            "Site effects at this site",
            _site_effects(
                rois,
                harmonization.gamma_star,
                harmonization.delta_star,
                mean_before,
                mean_after,
                sd_before,
                sd_after,
                harmonization.pooled_sd,
            ),
        ),
        (
            "cross-site",
            "Cross-site comparison",
            _cross_site(rois, summaries, harmonization.pooled_sd, parameters),
        ),
        ("settings", "Run settings", _settings(state, parameters)),
        (
            "output",
            "Output",
            _output(
                output_file_name if harmonized_data_is_reliable(state) else None,
                covariate_file_name,
            ),
        ),
    ]
    nav = "".join(
        f'<a class="nav-link" href="#sec-{slug}">{escape(title)}</a>'
        for slug, title, _ in sections
    )
    body = "".join(
        f'<section class="section" id="sec-{slug}">'
        f'<h2 class="section-title">{escape(title)}</h2>{content}</section>'
        for slug, title, content in sections
    )
    return _page(
        f'{header}<nav class="nav">{nav}</nav><main class="container">{body}</main>'
    )


# ── Sections ────────────────────────────────────────────────────────────────


def _header(own_name: str, site_names: List[str], chips) -> str:
    chip_html = "".join(
        f'<div class="chip">{escape(label)} <b>{escape(value)}</b></div>'
        for label, value in chips
    )
    pills = "".join(
        f'<span class="pill{" pill-own" if name == own_name else ""}">'
        f"{escape(name)}</span>"
        for name in site_names
    )
    return (
        '<header class="page-header">'
        '<button class="theme-toggle" type="button" onclick="toggleTheme()">'
        "Toggle theme</button>"
        "<h1>Decentralized ComBat Harmonization</h1>"
        f"<p>Results for site <b>{escape(own_name)}</b></p>"
        f'<div class="chips">{chip_html}</div>'
        f'<div class="pills">{pills}</div>'
        "</header>"
    )


def _diagnostics(
    state: SiteState, summaries: CrossSiteSummaries, parameters: Dict[str, Any]
) -> str:
    regression = state.regression
    items = "".join(
        f'<li class="finding finding-{f.level}">'
        f'<span class="badge badge-{f.level}">{f.level}</span>'
        f"<div><b>{escape(f.title)}</b>"
        f"{f'<p>{escape(f.detail)}</p>' if f.detail else ''}"
        + (
            f'<p class="recommendation"><span>Recommendation</span>'
            f"{escape(f.recommendation)}</p>"
            if f.recommendation
            else ""
        )
        + "</div></li>"
        for f in collect_findings(state, summaries, parameters)
    )
    return (
        f'<ul class="findings">{items}</ul>'
        '<p class="muted">Design matrix condition number: '
        f"{_fmt(regression.design_condition)} "
        f"(rank {regression.design_rank} of {regression.design_columns}).</p>"
    )


def _site_effects(
    rois,
    gamma_star,
    delta_star,
    mean_before,
    mean_after,
    sd_before,
    sd_after,
    pooled_sd,
) -> str:
    intro = (
        '<p class="muted">ComBat standardizes each ROI by the pooled residual SD, '
        "then estimates this site's additive shift (γ*) and multiplicative scale "
        "(δ*) with non-parametric empirical Bayes. γ* = 0 and δ* = 1 mean no site "
        "effect.</p>"
    )
    gamma_chart = _row_chart(
        rois,
        [_Series("γ*", gamma_star, "s-accent")],
        reference=0.0,
        axis_label="Additive effect γ* (pooled-SD units)",
    )
    delta_chart = _row_chart(
        rois,
        [_Series("δ*", delta_star, "s-accent")],
        reference=1.0,
        axis_label="Multiplicative effect δ* (variance ratio)",
    )
    rows = "".join(
        "<tr>"
        f"<td>{escape(roi)}</td>"
        f"<td>{_fmt(gamma_star[i], 3)}</td><td>{_fmt(delta_star[i], 3)}</td>"
        f"<td>{_fmt(mean_before[i])}</td><td>{_fmt(mean_after[i])}</td>"
        f"<td>{_fmt(sd_before[i])}</td><td>{_fmt(sd_after[i])}</td>"
        f"<td>{_fmt(pooled_sd[i])}</td>"
        "</tr>"
        for i, roi in enumerate(rois)
    )
    table = _table(
        [
            "ROI",
            "γ*",
            "δ*",
            "Mean before",
            "Mean after",
            "SD before",
            "SD after",
            "Pooled SD",
        ],
        rows,
    )
    return (
        f"{intro}"
        f'<div class="chart-grid"><div class="card">{gamma_chart}</div>'
        f'<div class="card">{delta_chart}</div></div>'
        f'<div class="card">{table}</div>'
    )


def _cross_site(
    rois,
    summaries: CrossSiteSummaries,
    pooled_sd,
    parameters: Dict[str, Any],
) -> str:
    if not summaries.enabled:
        shared = parameters.get("share_site_summaries", DEFAULT_SHARE_SITE_SUMMARIES)
        return (
            '<p class="note">Cross-site comparison is disabled '
            f"(<code>share_site_summaries</code> is <code>{str(shared).lower()}"
            "</code>). When enabled, each site shares per-ROI means and SDs before "
            "and after harmonization, and this section compares sites.</p>"
        )

    min_n = parameters.get("min_site_summary_n", DEFAULT_MIN_SITE_SUMMARY_N)
    parts = []
    if summaries.withheld_sites:
        parts.append(
            f'<p class="note">Sites with fewer than {min_n} subjects did not share '
            "summaries and are not shown: "
            f"{escape(', '.join(summaries.withheld_sites))}.</p>"
        )
    site_names = sorted(summaries.sites)
    if len(site_names) < 2:
        parts.append(
            '<p class="note">Fewer than two sites shared summaries, so there is '
            "nothing to compare.</p>"
        )
        return "".join(parts)

    spread_before, spread_after = between_site_spread(summaries, pooled_sd)
    parts.append(
        '<p class="muted">Between-site spread is the SD of the site means for each '
        "ROI, in pooled-SD units. Harmonization should move it toward 0.</p>"
    )
    parts.append(
        '<div class="card">'
        + _row_chart(
            rois,
            [
                _Series("Before", spread_before, "s-muted"),
                _Series("After", spread_after, "s-accent"),
            ],
            reference=0.0,
            axis_label="Between-site SD of site means (pooled-SD units)",
            connect=True,
        )
        + "</div>"
    )

    sample_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{summaries.sites[name].sample_count}</td></tr>"
        for name in site_names
    )
    headers = ["ROI"]
    for name in site_names:
        headers += [f"{name} before", f"{name} after"]
    mean_rows = "".join(
        f"<tr><td>{escape(roi)}</td>"
        + "".join(
            f"<td>{_fmt(summaries.sites[name].mean_before[i])}</td>"
            f"<td>{_fmt(summaries.sites[name].mean_after[i])}</td>"
            for name in site_names
        )
        + "</tr>"
        for i, roi in enumerate(rois)
    )
    parts.append(
        '<div class="chart-grid">'
        f'<div class="card"><h3>Subjects per site</h3>'
        f"{_table(['Site', 'Subjects'], sample_rows)}</div></div>"
        f'<div class="card"><h3>Site means before and after harmonization</h3>'
        f"{_table(headers, mean_rows)}</div>"
    )
    return "".join(parts)


def _settings(state: SiteState, parameters: Dict[str, Any]) -> str:
    covariates_types = parameters.get("covariates_types", {})
    covariate_list = ", ".join(
        f"{name} ({kind})" for name, kind in covariates_types.items()
    )
    settings = [
        ("Algorithm", parameters.get("combat_algo", "—")),
        ("Covariate file", parameters.get("covariate_file", "—")),
        ("Data file", parameters.get("data_file", "—")),
        ("Covariates", covariate_list or "—"),
        ("Encoded design columns", ", ".join(str(c) for c in state.covariates.columns)),
        ("Categorical levels", _levels_summary(state, parameters)),
        (
            "Share site summaries",
            str(
                parameters.get("share_site_summaries", DEFAULT_SHARE_SITE_SUMMARIES)
            ).lower(),
        ),
        (
            "Minimum site size for sharing",
            parameters.get("min_site_summary_n", DEFAULT_MIN_SITE_SUMMARY_N),
        ),
    ]
    rows = "".join(
        f"<tr><td>{escape(label)}</td><td>{escape(str(value))}</td></tr>"
        for label, value in settings
    )
    html = f'<div class="card">{_table(["Setting", "Value"], rows, "kv")}</div>'

    interpolated = state.interpolated_counts or {}
    if interpolated:
        interp_rows = "".join(
            f"<tr><td>{escape(roi)}</td><td>{count}</td></tr>"
            for roi, count in interpolated.items()
        )
        html += (
            '<div class="card"><h3>Interpolated values</h3>'
            f"{_table(['ROI', 'Values interpolated'], interp_rows)}</div>"
        )
    return html


def _levels_summary(state: SiteState, parameters: Dict[str, Any]) -> str:
    declared = parameters.get("categorical_levels") or {}
    parts = []
    for name, levels in state.layout.category_levels.items():
        if not levels:
            continue
        coded = [f"{levels[0]} (reference)"] + list(levels[1:])
        source = "declared" if name in declared else "from site data"
        parts.append(f"{name}: {', '.join(coded)} [{source}]")
    return "; ".join(parts) or "—"


def _output(output_file_name: Optional[str], covariate_file_name: Optional[str]) -> str:
    if output_file_name is None:
        parts = [
            '<p class="note">No harmonized data was written. The design matrix is '
            "rank-deficient, so the harmonized values would be unreliable. Follow "
            "the recommendation under Diagnostics and recommendations, then "
            "rerun.</p>"
        ]
    else:
        name = escape(output_file_name)
        parts = [
            "<p>The harmonized data has the same columns as the input data file, "
            "with site effects removed and values on the original measurement "
            "scale.</p>",
            f'<p><a class="download" href="{name}" download>Download {name}</a></p>',
        ]
    if covariate_file_name is not None:
        name = escape(covariate_file_name)
        parts.append(
            "<p>A copy of this site's covariate file is included for convenience; "
            "its rows are in the same order as the harmonized data. It stays at "
            "this site and is not shared.</p>"
            f'<p><a class="download" href="{name}" download>Download {name}</a></p>'
        )
    return f'<div class="card">{"".join(parts)}</div>'


# ── Building blocks ─────────────────────────────────────────────────────────


def _table(headers: List[str], rows_html: str, css_class: str = "") -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    return (
        f'<div class="table-scroll"><table class="stat-table {css_class}">'
        f"<thead><tr>{head}</tr></thead><tbody>{rows_html}</tbody></table></div>"
    )


def _row_chart(
    labels: Sequence[str],
    series: List[_Series],
    reference: Optional[float],
    axis_label: str,
    connect: bool = False,
) -> str:
    """Return an SVG dot plot with one row per label and a vertical reference."""
    row_height = 20
    top = 44 if len(series) > 1 else 30
    label_width = min(240, 16 + 6.4 * max((len(label) for label in labels), default=4))
    plot_width = 420
    right = 24
    width = label_width + plot_width + right
    height = top + row_height * len(labels) + 12

    finite = [
        float(v)
        for s in series
        for v in s.values
        if v is not None and math.isfinite(float(v))
    ]
    if reference is not None:
        finite.append(reference)
    low, high = (min(finite), max(finite)) if finite else (0.0, 1.0)
    if high - low < 1e-12:
        low, high = low - 1.0, high + 1.0
    padding = (high - low) * 0.06
    low, high = low - padding, high + padding

    def x(value: float) -> float:
        return label_width + (value - low) / (high - low) * plot_width

    parts = [
        f'<svg class="chart" viewBox="0 0 {width:.0f} {height}" '
        f'style="max-width:{width:.0f}px" '
        f'role="img" aria-label="{escape(axis_label)}">',
        f'<text class="axis-title" x="{label_width}" y="14">{escape(axis_label)}</text>',
    ]
    if len(series) > 1:
        legend_x = label_width
        for s in series:
            parts.append(
                f'<circle class="{s.css_class}" cx="{legend_x + 5}" cy="29" r="4.5"/>'
                f'<text class="legend" x="{legend_x + 14}" y="33">{escape(s.name)}</text>'
            )
            legend_x += 24 + 7 * len(s.name)

    for tick in _nice_ticks(low, high):
        tx = x(tick)
        parts.append(
            f'<line class="grid" x1="{tx:.1f}" y1="{top - 4}" x2="{tx:.1f}" '
            f'y2="{height - 12}"/>'
            f'<text class="tick" x="{tx:.1f}" y="{height - 1}">{_fmt(tick, 3)}</text>'
        )
    if reference is not None:
        rx = x(reference)
        parts.append(
            f'<line class="reference" x1="{rx:.1f}" y1="{top - 4}" x2="{rx:.1f}" '
            f'y2="{height - 12}"/>'
        )

    for row, label in enumerate(labels):
        cy = top + row_height * row + row_height / 2
        parts.append(
            f'<text class="row-label" x="{label_width - 8}" y="{cy + 4:.1f}">'
            f"{escape(label)}</text>"
        )
        points = []
        for s in series:
            value = s.values[row]
            if value is None or not math.isfinite(float(value)):
                points.append(None)
                continue
            points.append((x(float(value)), s, float(value)))
        drawn = [p for p in points if p is not None]
        if connect and len(drawn) > 1:
            parts.append(
                f'<line class="connector" x1="{drawn[0][0]:.1f}" y1="{cy:.1f}" '
                f'x2="{drawn[-1][0]:.1f}" y2="{cy:.1f}"/>'
            )
        elif not connect and reference is not None and drawn:
            parts.append(
                f'<line class="stem" x1="{x(reference):.1f}" y1="{cy:.1f}" '
                f'x2="{drawn[0][0]:.1f}" y2="{cy:.1f}"/>'
            )
        for cx, s, value in drawn:
            parts.append(
                f'<circle class="{s.css_class}" cx="{cx:.1f}" cy="{cy:.1f}" r="4.5">'
                f"<title>{escape(label)} · {escape(s.name)}: {_fmt(value, 4)}</title>"
                "</circle>"
            )

    parts.append("</svg>")
    return "".join(parts)


def _nice_ticks(low: float, high: float, count: int = 5) -> List[float]:
    raw_step = (high - low) / count
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw_step)
    start = math.ceil(low / step) * step
    ticks = []
    value = start
    while value <= high + 1e-12:
        ticks.append(0.0 if abs(value) < step * 1e-9 else value)
        value += step
    return ticks


def _fmt(value, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(value):
        return "—" if math.isnan(value) else "∞"
    return f"{value:.{digits}g}"


def _page(body: str) -> str:
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>'
        '<meta name="viewport" content="width=device-width,initial-scale=1"/>'
        "<title>ComBat Harmonization Report</title>"
        f"<style>{_CSS}</style></head><body>{body}<script>{_JS}</script>"
        "</body></html>"
    )


_CSS = """
:root{--bg:#fff;--bg3:#f1f5f9;--border:#e2e8f0;--border2:#cbd5e1;--text:#0f172a;
--text2:#334155;--text3:#64748b;--header-bg:linear-gradient(135deg,#e0e9ff 0%,#f8fafc 100%);
--chip-bg:#f1f5f9;--chip-b:#4f46e5;--card-bg:#fff;--th-bg:#f8fafc;--accent:#4f46e5;
--muted-dot:#94a3b8;--own-bg:rgba(79,70,229,.12);--note-bg:#f8fafc}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f172a;
--bg3:#1a2640;--border:#334155;--border2:#475569;--text:#e2e8f0;--text2:#cbd5e1;
--text3:#94a3b8;--header-bg:linear-gradient(135deg,#1e1b4b 0%,#0f172a 100%);
--chip-bg:#1e293b;--chip-b:#a5b4fc;--card-bg:#1e293b;--th-bg:#161f30;--accent:#a5b4fc;
--muted-dot:#64748b;--own-bg:rgba(165,180,252,.15);--note-bg:#162033}}
:root[data-theme="dark"]{--bg:#0f172a;--bg3:#1a2640;--border:#334155;--border2:#475569;
--text:#e2e8f0;--text2:#cbd5e1;--text3:#94a3b8;
--header-bg:linear-gradient(135deg,#1e1b4b 0%,#0f172a 100%);--chip-bg:#1e293b;
--chip-b:#a5b4fc;--card-bg:#1e293b;--th-bg:#161f30;--accent:#a5b4fc;--muted-dot:#64748b;
--own-bg:rgba(165,180,252,.15);--note-bg:#162033}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:var(--bg);color:var(--text);line-height:1.5}
.page-header{background:var(--header-bg);border-bottom:1px solid var(--border);
padding:2rem 1rem;position:relative}
.page-header>*{max-width:1200px;margin-left:auto;margin-right:auto}
.page-header h1{font-size:1.6rem;font-weight:700;letter-spacing:-.02em;padding-right:8rem}
.page-header p{color:var(--text3);margin-top:.3rem}
.theme-toggle{position:absolute;top:1.5rem;right:1rem;background:var(--card-bg);
border:1px solid var(--border);border-radius:999px;padding:.3rem .8rem;font-size:.78rem;
font-weight:600;color:var(--text2);cursor:pointer}
.chips,.pills{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1rem}
.chip{background:var(--chip-bg);border:1px solid var(--border);border-radius:999px;
padding:.25rem .75rem;font-size:.78rem;color:var(--text2)}
.chip b{color:var(--chip-b)}
.pill{border:1px solid var(--border2);border-radius:999px;padding:.15rem .6rem;
font-size:.75rem;color:var(--text2)}
.pill-own{background:var(--own-bg);border-color:var(--accent);color:var(--accent);
font-weight:600}
.nav{display:flex;gap:.4rem;flex-wrap:wrap;max-width:1200px;margin:1rem auto 0;
padding:0 1rem}
.nav-link{font-size:.8rem;color:var(--text3);text-decoration:none;padding:.3rem .7rem;
border-radius:8px}
.nav-link:hover{background:var(--bg3);color:var(--text)}
.container{max-width:1200px;margin:0 auto;padding:1.5rem 1rem 3rem}
.section{margin-bottom:2.5rem;scroll-margin-top:1rem}
.section-title{font-size:.8rem;font-weight:700;text-transform:uppercase;
letter-spacing:.08em;color:var(--text3);margin-bottom:1rem;padding-bottom:.5rem;
border-bottom:1px solid var(--border)}
h3{font-size:.9rem;margin-bottom:.6rem}
.muted{color:var(--text3);font-size:.85rem;margin-bottom:1rem}
.note{background:var(--note-bg);border:1px solid var(--border);border-radius:10px;
padding:.75rem 1rem;font-size:.88rem;color:var(--text2);margin-bottom:1rem}
.card{background:var(--card-bg);border:1px solid var(--border);border-radius:12px;
padding:1rem;margin-bottom:1rem;min-width:0}
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));
gap:1rem}
.chart{width:100%;height:auto;display:block}
.chart text{fill:var(--text3);font-size:11px}
.chart .row-label{text-anchor:end;fill:var(--text2)}
.chart .axis-title{font-weight:600;fill:var(--text2)}
.chart .tick{text-anchor:middle}
.chart .grid{stroke:var(--border);stroke-width:1}
.chart .reference{stroke:var(--border2);stroke-width:1.5;stroke-dasharray:4 3}
.chart .stem,.chart .connector{stroke:var(--border2);stroke-width:2}
.chart .s-accent{fill:var(--accent)}
.chart .s-muted{fill:var(--muted-dot)}
.table-scroll{overflow-x:auto}
table.stat-table{width:100%;border-collapse:collapse;font-size:.8rem}
.stat-table th{color:var(--text3);font-weight:600;padding:.45rem .75rem;text-align:right;
background:var(--th-bg);white-space:nowrap}
.stat-table td{padding:.4rem .75rem;border-top:1px solid var(--border);text-align:right;
font-variant-numeric:tabular-nums}
.stat-table th:first-child,.stat-table td:first-child{text-align:left}
.stat-table.kv td:last-child{text-align:left}
.findings{list-style:none;display:grid;gap:.6rem;margin-bottom:.8rem}
.finding{display:flex;gap:.75rem;align-items:flex-start;background:var(--card-bg);
border:1px solid var(--border);border-radius:10px;padding:.75rem 1rem;font-size:.88rem}
.finding p{color:var(--text3);margin-top:.2rem}
.badge{flex-shrink:0;min-width:5.2rem;text-align:center;border-radius:999px;padding:.1rem .55rem;font-size:.7rem;
font-weight:700;text-transform:uppercase;letter-spacing:.04em;border:1px solid}
.badge-ok{color:#15803d;background:rgba(16,185,129,.12);border-color:rgba(16,185,129,.35)}
.badge-info{color:#1d4ed8;background:rgba(59,130,246,.1);border-color:rgba(59,130,246,.3)}
.badge-warning{color:#b45309;background:rgba(245,158,11,.12);border-color:rgba(245,158,11,.35)}
.badge-error{color:#b91c1c;background:rgba(239,68,68,.12);border-color:rgba(239,68,68,.35)}
.finding-error{border-color:rgba(239,68,68,.45)}
.recommendation{margin-top:.45rem;padding:.5rem .7rem;border-radius:8px;
background:var(--bg3);color:var(--text2)!important}
.recommendation span{display:block;font-size:.7rem;font-weight:700;text-transform:uppercase;
letter-spacing:.05em;color:var(--accent);margin-bottom:.1rem}
.download{color:var(--accent);font-weight:600}
code{font-size:.85em}
@media (max-width:600px){.page-header h1{font-size:1.3rem;padding-right:0}
.theme-toggle{position:static;margin-bottom:.8rem}}
"""

_JS = """
function toggleTheme(){var r=document.documentElement;
var dark=r.getAttribute('data-theme')==='dark'||(!r.getAttribute('data-theme')&&
window.matchMedia('(prefers-color-scheme: dark)').matches);
var next=dark?'light':'dark';r.setAttribute('data-theme',next);
try{localStorage.setItem('combat-theme',next)}catch(e){}}
(function(){try{var t=localStorage.getItem('combat-theme');
if(t)document.documentElement.setAttribute('data-theme',t)}catch(e){}})();
"""
