"""Comparison page: artist separation across models."""

import dash
from dash import html, dcc, Input, Output, callback, dash_table, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from components.data_loader import get_comparison, get_model
from components.translations import t
from components.colors import MODEL_COLORS, TOPIC_COLORS, topic_color

dash.register_page(__name__, path="/compare-artists", name="Artists")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_KEYS = ["bertopic", "lda", "iramuteq"]
MODEL_DISPLAY = {"bertopic": "BERTopic", "lda": "LDA", "iramuteq": "IRAMUTEQ"}

METRIC_REF_CRAMERS = {
    "citation": "Cramer, H. (1946). Mathematical Methods of Statistics. Princeton University Press.",
    "desc_fr": "Le V de Cramer mesure la force d'association entre deux variables categorielles. V = sqrt(chi2 / (n x min(k-1, r-1))). Intervalle : [0, 1], ou 0 = aucune association, 1 = association parfaite.",
    "desc_en": "Cramer's V measures the strength of association between two categorical variables. V = sqrt(chi2 / (n x min(k-1, r-1))). Range: [0, 1], where 0 = no association, 1 = perfect association.",
}


def _fmt(value, fmt=".3f"):
    if value is None:
        return "N/A"
    try:
        return f"{value:{fmt}}"
    except (TypeError, ValueError):
        return str(value)


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
# Cramer's V bar chart
# ---------------------------------------------------------------------------

def _cramers_v_chart(comparison, lang):
    artist_sep = comparison.get("artist_separation", {})

    names, values, colors = [], [], []
    for mk in MODEL_KEYS:
        display = MODEL_DISPLAY[mk]
        entry = artist_sep.get(mk, {})
        v = entry.get("cramers_v")
        if v is not None:
            names.append(display)
            values.append(v)
            colors.append(MODEL_COLORS.get(display, "#999"))

    if not values:
        return _empty_fig(lang)

    fig = go.Figure(go.Bar(
        x=names, y=values, marker_color=colors,
        text=[f"{v:.3f}" for v in values], textposition="outside",
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=t("cramers_v", lang),
        yaxis_title=t("cramers_v", lang),
        margin=dict(l=60, r=20, t=50, b=40),
        plot_bgcolor="white", height=380,
        yaxis=dict(gridcolor="#eee", rangemode="tozero"),
        xaxis=dict(gridcolor="#eee"),
    )
    return fig


# ---------------------------------------------------------------------------
# Standardized residuals heatmaps
# ---------------------------------------------------------------------------

def _residuals_heatmap(model_key, comparison, lang):
    z_data = comparison.get("residuals", {}).get(model_key)
    labels_data = comparison.get("residuals_labels", {}).get(model_key, {})

    if z_data is None:
        return _empty_fig(lang)

    row_labels = labels_data.get("rows", [str(i) for i in range(len(z_data))])
    col_labels = labels_data.get("cols", [str(j) for j in range(len(z_data[0]) if z_data else [])])

    flat = [v for row in z_data for v in row if v is not None]
    abs_max = max(abs(min(flat)), abs(max(flat))) if flat else 1

    fig = go.Figure(go.Heatmap(
        z=z_data, x=col_labels, y=row_labels,
        colorscale="RdBu_r", zmid=0, zmin=-abs_max, zmax=abs_max,
        hovertemplate=(
            t("artist", lang) + ": %{y}<br>"
            + t("topic", lang) + ": %{x}<br>"
            + t("residuals", lang) + ": %{z:.2f}<extra></extra>"
        ),
        colorbar=dict(title=t("residuals", lang)),
    ))

    display_name = MODEL_DISPLAY.get(model_key, model_key)
    fig.update_layout(
        title=f"{t('residuals', lang)} - {display_name}",
        xaxis_title=t("topic", lang), yaxis_title=t("artist", lang),
        margin=dict(l=100, r=20, t=50, b=80),
        plot_bgcolor="white",
        height=max(450, len(row_labels) * 18),
        yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
        xaxis=dict(tickfont=dict(size=10)),
    )
    return fig


# ---------------------------------------------------------------------------
# Artist comparison across models (fixed topic display)
# ---------------------------------------------------------------------------

def _all_artists_options():
    all_artists = set()
    for mk in MODEL_KEYS:
        model_data = get_model(mk)
        artist_metrics = model_data.get("artist_metrics", [])
        for am in artist_metrics:
            name = am.get("artist", "")
            if name:
                all_artists.add(name)
    return sorted(all_artists)


def _parse_top_topics(entry, labels):
    """Parse top_3_topics and top_3_ratios (comma-separated strings or lists)."""
    top_topics_raw = entry.get("top_3_topics", entry.get("top_topics", []))
    top_ratios_raw = entry.get("top_3_ratios", entry.get("top_ratios", []))

    if isinstance(top_topics_raw, str):
        topic_ids = [x.strip() for x in top_topics_raw.split(",") if x.strip()]
    elif isinstance(top_topics_raw, list):
        topic_ids = top_topics_raw
    else:
        topic_ids = []

    if isinstance(top_ratios_raw, str):
        try:
            ratios = [float(x.strip()) for x in top_ratios_raw.split(",") if x.strip()]
        except ValueError:
            ratios = []
    elif isinstance(top_ratios_raw, list):
        ratios = top_ratios_raw
    else:
        ratios = []

    items = []
    for i, tid in enumerate(topic_ids[:5]):
        if isinstance(tid, dict):
            t_id = tid.get("topic", tid.get("topic_id", ""))
            pct = tid.get("pct", tid.get("proportion", 0))
            lbl = labels.get(str(t_id), f"T{t_id}")
            pct_str = f"{pct:.1%}" if isinstance(pct, float) else str(pct)
            items.append(html.Li(f"{lbl} ({pct_str})", className="small"))
        else:
            lbl = labels.get(str(tid), f"T{tid}")
            if i < len(ratios) and isinstance(ratios[i], (int, float)):
                pct_str = f"{ratios[i]:.1%}"
                items.append(html.Li(f"{lbl} ({pct_str})", className="small"))
            else:
                items.append(html.Li(lbl, className="small"))

    return items


def _artist_comparison_cards(artist_name, lang):
    if not artist_name:
        return html.P(t("select_artist", lang), className="text-muted")

    cols = []
    for mk in MODEL_KEYS:
        model_data = get_model(mk)
        display_name = MODEL_DISPLAY[mk]
        labels = model_data.get("topic_labels", {})
        artist_metrics = model_data.get("artist_metrics", [])

        entry = None
        for am in artist_metrics:
            if am.get("artist") == artist_name:
                entry = am
                break

        if entry is None:
            cols.append(dbc.Col(
                dbc.Card([
                    dbc.CardHeader(
                        html.H6(display_name, className="mb-0"),
                        style={"backgroundColor": MODEL_COLORS.get(display_name, "#6c757d"), "color": "white"},
                    ),
                    dbc.CardBody(html.P(t("no_data", lang), className="text-muted")),
                ], className="h-100 shadow-sm"),
                md=4,
            ))
            continue

        dominant = entry.get("dominant_topic", "N/A")
        dominant_label = labels.get(str(dominant), f"T{dominant}")
        n_docs = entry.get("n_docs", 0)

        top_items = _parse_top_topics(entry, labels)

        card = dbc.Card([
            dbc.CardHeader(
                html.H6(display_name, className="mb-0"),
                style={"backgroundColor": MODEL_COLORS.get(display_name, "#6c757d"), "color": "white"},
            ),
            dbc.CardBody([
                html.P([
                    html.Strong(t("dominant_topic", lang) + ": "),
                    html.Span(dominant_label, className="small"),
                ]),
                html.P([
                    html.Strong(t("n_docs", lang) + ": "),
                    html.Span(f"{n_docs:,}"),
                ]),
                html.P(html.Strong(t("top_topics", lang) + ":"), className="mb-1"),
                html.Ul(top_items) if top_items else html.P("-", className="text-muted small"),
            ]),
        ], className="h-100 shadow-sm")

        cols.append(dbc.Col(card, md=4))

    return dbc.Row(cols, className="g-3")


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

    # Section 1: Cramer's V
    cramers_section = html.Div([
        html.H4(t("cramers_v", lang), className="section-header"),
        html.Div(
            dcc.Graph(
                figure=_cramers_v_chart(comparison, lang),
                config={"displayModeBar": False},
            ),
            className="chart-container",
        ),
        # Metric footnote
        html.Div(
            html.Small(
                f"* {t('cramers_v', lang)} : "
                f"{METRIC_REF_CRAMERS.get(f'desc_{lang}', METRIC_REF_CRAMERS['desc_en'])} "
                f"({METRIC_REF_CRAMERS['citation']})",
                className="text-muted",
            ),
            className="metric-footnote",
        ),
    ])

    # Section 2: Residuals heatmaps (3 tabs)
    residuals_tabs = dbc.Tabs([
        dbc.Tab(
            html.Div(
                dcc.Graph(
                    figure=_residuals_heatmap(mk, comparison, lang),
                    config={"displayModeBar": True, "displaylogo": False},
                ),
                className="chart-container",
            ),
            label=MODEL_DISPLAY[mk],
            tab_id=f"residuals-tab-{mk}",
        )
        for mk in MODEL_KEYS
    ], id="residuals-tabs", active_tab=f"residuals-tab-{MODEL_KEYS[0]}")

    residuals_section = html.Div([
        html.H4(t("residuals", lang), className="section-header"),
        residuals_tabs,
    ])

    # Section 3: Artist search
    artist_options = [{"label": a, "value": a} for a in _all_artists_options()]
    artist_section = html.Div([
        html.H4(t("artist_explorer", lang), className="section-header"),
        dbc.Row([
            dbc.Col([
                html.Label(t("select_artist", lang), className="fw-bold mb-1"),
                dcc.Dropdown(
                    id="compare-artist-dropdown",
                    options=artist_options, value=None,
                    placeholder=t("search_placeholder", lang),
                    searchable=True, clearable=True, className="mb-3",
                ),
            ], md=6),
        ]),
        html.Div(
            html.P(t("select_artist", lang), className="text-muted"),
            id="compare-artist-cards",
        ),
    ])

    footer = html.Div(t("footer_text", lang), className="footer")

    return html.Div(
        [
            html.H2(t("artists", lang), className="section-header"),
            cramers_section,
            residuals_section,
            artist_section,
            footer,
        ],
        className="page-container",
    )


# ---------------------------------------------------------------------------
# Page wiring
# ---------------------------------------------------------------------------

layout = html.Div(id="compare-artists-content")


@callback(Output("compare-artists-content", "children"), Input("lang-store", "data"))
def update(lang):
    lang = lang or "fr"
    return build_layout(lang)


@callback(
    Output("compare-artist-cards", "children"),
    Input("compare-artist-dropdown", "value"),
    Input("lang-store", "data"),
    prevent_initial_call=True,
)
def update_artist_comparison(artist_name, lang):
    lang = lang or "fr"
    return _artist_comparison_cards(artist_name, lang)
