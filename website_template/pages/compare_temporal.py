"""Comparison page: temporal dynamics across models."""

import dash
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components.data_loader import get_comparison, get_model
from components.translations import t
from components.colors import MODEL_COLORS, TOPIC_COLORS, topic_color

dash.register_page(__name__, path="/compare-temporal", name="Temporal")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_KEYS = ["bertopic", "lda", "iramuteq"]
MODEL_DISPLAY = {"bertopic": "BERTopic", "lda": "LDA", "iramuteq": "IRAMUTEQ"}


def _empty_fig(lang, height=350):
    fig = go.Figure()
    fig.add_annotation(
        text=t("no_data", lang),
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color="#999"),
    )
    fig.update_layout(
        height=height, margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="white", xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Mean temporal variance bar chart
# ---------------------------------------------------------------------------

def _mean_variance_chart(comparison, lang):
    temporal = comparison.get("temporal", {})

    names, values, colors = [], [], []
    for mk in MODEL_KEYS:
        display = MODEL_DISPLAY[mk]
        entry = temporal.get(mk, {})
        mv = entry.get("mean_variance")
        if mv is not None:
            names.append(display)
            values.append(mv)
            colors.append(MODEL_COLORS.get(display, "#999"))

    if not values:
        return _empty_fig(lang)

    fig = go.Figure(go.Bar(
        x=names, y=values, marker_color=colors,
        text=[f"{v:.5f}" for v in values], textposition="outside",
        hovertemplate="%{x}: %{y:.6f}<extra></extra>",
    ))
    fig.update_layout(
        title=t("mean_variance", lang),
        yaxis_title=t("mean_variance", lang),
        margin=dict(l=70, r=20, t=50, b=40),
        plot_bgcolor="white", height=380,
        yaxis=dict(gridcolor="#eee", rangemode="tozero", exponentformat="e"),
        xaxis=dict(gridcolor="#eee"),
    )
    return fig


# ---------------------------------------------------------------------------
# Side-by-side stacked area subplots (fixed legend)
# ---------------------------------------------------------------------------

def _stacked_area_subplots(lang):
    titles = [MODEL_DISPLAY[mk] for mk in MODEL_KEYS]
    fig = make_subplots(
        rows=1, cols=3, subplot_titles=titles,
        shared_yaxes=True, horizontal_spacing=0.05,
    )

    any_data = False
    n_topics = 0
    for col_idx, mk in enumerate(MODEL_KEYS, start=1):
        model_data = get_model(mk)
        evolution = model_data.get("topic_evolution", [])
        labels = model_data.get("topic_labels", {})

        if not evolution:
            fig.add_annotation(
                text=t("no_data", lang),
                xref=f"x{col_idx}", yref=f"y{col_idx}",
                x=0.5, y=0.5,
                showarrow=False, font=dict(size=14, color="#999"),
            )
            continue

        any_data = True
        years = [row.get("year", row.get("period", "")) for row in evolution]
        topic_ids = sorted(
            [k for k in evolution[0].keys() if k not in ("year", "period")],
            key=lambda x: int(x) if str(x).isdigit() else 0,
        )
        n_topics = max(n_topics, len(topic_ids))

        for tid in topic_ids:
            vals = [row.get(tid, row.get(str(tid), 0)) for row in evolution]
            color = topic_color(tid)
            label = labels.get(str(tid), f"T{tid}")
            fig.add_trace(
                go.Scatter(
                    x=years, y=vals, mode="lines",
                    name=label, stackgroup="one",
                    line=dict(width=0.5, color=color),
                    fillcolor=color,
                    hovertemplate=f"{label}<br>{t('year', lang)}: %{{x}}<br>{t('proportion', lang)}: %{{y:.3f}}<extra></extra>",
                    showlegend=(col_idx == 1),
                    legendgroup=str(tid),
                ),
                row=1, col=col_idx,
            )

    if not any_data:
        return _empty_fig(lang, height=450)

    # Compute dynamic bottom margin based on number of topics
    n_cols_legend = min(5, n_topics)
    n_rows_legend = (n_topics + n_cols_legend - 1) // n_cols_legend if n_cols_legend > 0 else 1
    bottom_margin = max(80, 40 + 18 * n_rows_legend)

    fig.update_layout(
        height=550,
        margin=dict(l=50, r=20, t=50, b=bottom_margin),
        plot_bgcolor="white",
        legend=dict(
            font=dict(size=8),
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            tracegroupgap=2,
        ),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#eee", tickangle=-45, tickfont=dict(size=9), row=1, col=i)
        fig.update_yaxes(gridcolor="#eee", row=1, col=i)
    fig.update_yaxes(title_text=t("proportion", lang), row=1, col=1)

    return fig


# ---------------------------------------------------------------------------
# Decade JS divergence grouped bar chart
# ---------------------------------------------------------------------------

def _decade_js_chart(lang):
    decades_per_model = {}
    has_data = False

    for mk in MODEL_KEYS:
        model_data = get_model(mk)
        metrics = model_data.get("metrics", {})
        temporal_metrics = metrics.get("temporal_metrics", {})
        decade_changes = temporal_metrics.get("decade_changes", [])

        if decade_changes:
            has_data = True
            decades_per_model[mk] = decade_changes

    if not has_data:
        return None

    all_decades = set()
    for mk, changes in decades_per_model.items():
        for entry in changes:
            decade_label = entry.get("decade", entry.get("period", ""))
            if decade_label:
                all_decades.add(str(decade_label))
    all_decades = sorted(all_decades)

    if not all_decades:
        return None

    fig = go.Figure()
    for mk in MODEL_KEYS:
        display = MODEL_DISPLAY[mk]
        changes = decades_per_model.get(mk, [])

        decade_js = {}
        for entry in changes:
            decade_label = str(entry.get("decade", entry.get("period", "")))
            js_val = entry.get("js_divergence", entry.get("js_div", entry.get("mean_js", 0)))
            decade_js[decade_label] = js_val

        vals = [decade_js.get(d, 0) for d in all_decades]
        fig.add_trace(go.Bar(
            name=display, x=all_decades, y=vals,
            marker_color=MODEL_COLORS.get(display, "#999"),
            hovertemplate=display + "<br>" + t("decade", lang) + ": %{x}<br>JS: %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        title=t("js_divergence", lang) + " / " + t("decade", lang),
        xaxis_title=t("decade", lang),
        yaxis_title=t("js_divergence", lang),
        barmode="group",
        margin=dict(l=60, r=20, t=50, b=60),
        plot_bgcolor="white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        yaxis=dict(gridcolor="#eee", rangemode="tozero"),
        xaxis=dict(gridcolor="#eee"),
    )
    return fig


# ---------------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------------

def build_layout(lang):
    comparison = get_comparison()
    if not comparison:
        return html.Div(
            html.P(t("no_data", lang), className="text-muted p-5 text-center"),
            className="page-container",
        )

    # Section 1: Mean temporal variance
    variance_section = html.Div([
        html.H4(t("mean_variance", lang), className="section-header"),
        html.Div(
            dcc.Graph(
                figure=_mean_variance_chart(comparison, lang),
                config={"displayModeBar": False},
            ),
            className="chart-container",
        ),
    ])

    # Section 2: Side-by-side stacked area
    area_section = html.Div([
        html.H4(t("topic_evolution", lang), className="section-header"),
        html.Div(
            dcc.Graph(
                figure=_stacked_area_subplots(lang),
                config={"displayModeBar": True, "displaylogo": False},
            ),
            className="chart-container",
        ),
    ])

    children = [
        html.H2(t("temporal", lang), className="section-header"),
        variance_section,
        area_section,
    ]

    # Section 3: Decade JS divergence (optional)
    decade_fig = _decade_js_chart(lang)
    if decade_fig is not None:
        decade_section = html.Div([
            html.H4(t("js_divergence", lang) + " / " + t("decade", lang), className="section-header"),
            html.Div(
                dcc.Graph(figure=decade_fig, config={"displayModeBar": False}),
                className="chart-container",
            ),
        ])
        children.append(decade_section)

    # Metric footnote
    children.append(
        html.Div(
            html.Small(
                "* " + t("js_divergence", lang) + " : "
                + (
                    "La divergence JS mesure la dissimilarite entre distributions de probabilite des topics entre deux periodes. Plus la valeur est elevee, plus la distribution des topics a change entre les periodes comparees. Intervalle : [0, 1]. "
                    if lang == "fr" else
                    "JS divergence measures the dissimilarity between topic probability distributions across two time periods. Higher values indicate greater change in topic distribution between compared periods. Range: [0, 1]. "
                )
                + "(Lin, J. (1991). Divergence measures based on the Shannon entropy. IEEE Trans. Information Theory, 37(1), 145-151.)",
                className="text-muted",
            ),
            className="metric-footnote",
        ),
    )

    children.append(html.Div(t("footer_text", lang), className="footer"))

    return html.Div(children, className="page-container")


# ---------------------------------------------------------------------------
# Page wiring
# ---------------------------------------------------------------------------

layout = html.Div(id="compare-temporal-content")


@callback(Output("compare-temporal-content", "children"), Input("lang-store", "data"))
def update(lang):
    lang = lang or "fr"
    return build_layout(lang)
