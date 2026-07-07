"""Comparison page: topic distances and aggregation stability."""

import dash
from dash import html, dcc, dash_table, Input, Output, callback
import plotly.graph_objects as go

from components.data_loader import get_comparison
from components.translations import t
from components.colors import MODEL_COLORS

dash.register_page(__name__, path="/compare-distances", name="Distances")

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
# Aggregation stabilization curves
# ---------------------------------------------------------------------------

def _aggregation_curves(comparison, lang):
    multi_agg = comparison.get("multi_aggregation", {})
    if not multi_agg:
        return _empty_fig(lang, height=450)

    has_data = False
    fig = go.Figure()

    for mk in MODEL_KEYS:
        entry = multi_agg.get(mk, {})
        sizes = entry.get("aggregation_sizes", [])
        intra_means = entry.get("intra_labbe_means", [])
        inter_means = entry.get("inter_labbe_means", [])

        if not sizes:
            continue
        has_data = True

        display = MODEL_DISPLAY[mk]
        color = MODEL_COLORS.get(display, "#999")

        if intra_means:
            fig.add_trace(go.Scatter(
                x=sizes, y=intra_means,
                mode="lines+markers",
                name=f"{display} - {t('intra_distance', lang)}",
                line=dict(color=color, width=2, dash="dash"),
                marker=dict(size=6, symbol="circle"),
                legendgroup=mk,
                hovertemplate=f"{display} intra<br>n = %{{x}}<br>Labbe = %{{y:.4f}}<extra></extra>",
            ))

        if inter_means:
            fig.add_trace(go.Scatter(
                x=sizes, y=inter_means,
                mode="lines+markers",
                name=f"{display} - {t('inter_distance', lang)}",
                line=dict(color=color, width=2, dash="solid"),
                marker=dict(size=6, symbol="diamond"),
                legendgroup=mk,
                hovertemplate=f"{display} inter<br>n = %{{x}}<br>Labbe = %{{y:.4f}}<extra></extra>",
            ))

    if not has_data:
        return _empty_fig(lang, height=450)

    fig.update_layout(
        title=t("aggregation_curve", lang),
        xaxis_title="Aggregation size (n documents)" if lang == "en" else "Taille d'agregation (n documents)",
        yaxis_title="Labbe distance" if lang == "en" else "Distance de Labbe",
        margin=dict(l=70, r=20, t=50, b=80),
        plot_bgcolor="white", height=480,
        legend=dict(
            font=dict(size=10), orientation="h",
            yanchor="top", y=-0.18, xanchor="center", x=0.5,
        ),
        xaxis=dict(gridcolor="#eee"),
        yaxis=dict(gridcolor="#eee", rangemode="tozero"),
    )
    return fig


# ---------------------------------------------------------------------------
# Aggregation SR table
# ---------------------------------------------------------------------------

DISPLAY_SIZES = [8, 24, 72]

def _aggregation_sr_table(comparison, lang):
    """Build a DataTable with intra, inter, and SR for selected aggregation sizes."""
    multi_agg = comparison.get("multi_aggregation", {})
    if not multi_agg:
        return html.P(t("no_data", lang), className="text-muted")

    rows = []
    for mk in MODEL_KEYS:
        entry = multi_agg.get(mk, {})
        sizes = entry.get("aggregation_sizes", [])
        intra = entry.get("intra_labbe_means", [])
        inter = entry.get("inter_labbe_means", [])
        if not sizes:
            continue

        row = {t("model_comparison", lang): MODEL_DISPLAY[mk]}
        for n in DISPLAY_SIZES:
            if n in sizes:
                idx = sizes.index(n)
                intra_v = intra[idx] if idx < len(intra) else None
                inter_v = inter[idx] if idx < len(inter) else None
                sr_v = inter_v / intra_v if intra_v and inter_v and intra_v > 0 else None
                row[f"N={n} Intra"] = f"{intra_v:.4f}" if intra_v is not None else "—"
                row[f"N={n} Inter"] = f"{inter_v:.4f}" if inter_v is not None else "—"
                row[f"N={n} SR"] = f"{sr_v:.2f}" if sr_v is not None else "—"
        rows.append(row)

    if not rows:
        return html.P(t("no_data", lang), className="text-muted")

    columns = [{"name": col, "id": col} for col in rows[0].keys()]

    return dash_table.DataTable(
        data=rows,
        columns=columns,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#343a40",
            "color": "white",
            "fontWeight": "bold",
            "textAlign": "center",
        },
        style_cell={
            "textAlign": "center",
            "padding": "8px 12px",
            "fontSize": "0.85rem",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
        ],
    )


# ---------------------------------------------------------------------------
# D3-sem: semantic (embedding-space) evaluation
# ---------------------------------------------------------------------------

def _semantic_ch_chart(sem, lang):
    """Grouped bar chart: Calinski-Harabasz by evaluation space, per model."""
    spaces = sem.get("eval_spaces", [])
    per_space = sem.get("per_space", {})
    if not spaces:
        return _empty_fig(lang, height=400)

    fig = go.Figure()
    for mk in MODEL_KEYS:
        display = MODEL_DISPLAY[mk]
        ys = [per_space.get(sp, {}).get(mk, {}).get("cluster_quality", {}).get("calinski_harabasz", 0)
              for sp in spaces]
        fig.add_trace(go.Bar(
            x=spaces, y=ys, name=display,
            marker_color=MODEL_COLORS.get(display, "#999"),
            hovertemplate=f"{display}<br>%{{x}}<br>CH = %{{y:.1f}}<extra></extra>",
        ))

    clustered = sem.get("bertopic_clustered_space")
    if clustered in spaces:
        fig.add_annotation(
            x=clustered, y=1.0, yref="paper",
            text="⟳ " + t("semantic_circular_tag", lang), showarrow=False,
            font=dict(size=10, color="#b00"), yanchor="bottom",
        )

    fig.update_layout(
        title=t("semantic_ch_chart_title", lang),
        barmode="group", plot_bgcolor="white", height=420,
        margin=dict(l=70, r=20, t=50, b=50),
        xaxis_title=t("semantic_space", lang),
        yaxis_title="Calinski-Harabasz",
        yaxis=dict(gridcolor="#eee", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


def _semantic_cluster_quality_table(sem, lang):
    spaces = sem.get("eval_spaces", [])
    per_space = sem.get("per_space", {})
    if not spaces:
        return html.P(t("no_data", lang), className="text-muted")

    space_col = t("semantic_space", lang)
    model_col = t("model_comparison", lang)
    rows = []
    for sp in spaces:
        circ = per_space.get(sp, {}).get("bertopic", {}).get("cluster_quality", {}).get("circular")
        sp_label = sp + (f" (⟳ {t('semantic_circular_tag', lang)})" if circ else "")
        for mk in MODEL_KEYS:
            cq = per_space.get(sp, {}).get(mk, {}).get("cluster_quality", {})
            rows.append({
                space_col: sp_label if mk == "bertopic" else "",
                model_col: MODEL_DISPLAY[mk],
                t("silhouette_cosine", lang): f"{cq.get('silhouette_cosine', 0):.4f}",
                t("calinski_harabasz", lang): f"{cq.get('calinski_harabasz', 0):,.1f}",
                t("davies_bouldin", lang): f"{cq.get('davies_bouldin', 0):.4f}",
            })

    columns = [{"name": c, "id": c} for c in rows[0].keys()]
    return dash_table.DataTable(
        data=rows, columns=columns,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#343a40", "color": "white",
                      "fontWeight": "bold", "textAlign": "center"},
        style_cell={"textAlign": "center", "padding": "8px 12px", "fontSize": "0.85rem"},
        style_data_conditional=[
            {"if": {"filter_query": "{" + model_col + "} = BERTopic"},
             "backgroundColor": "#eef3fb"},
        ],
    )


def _semantic_sr_table(sem, lang):
    spaces = sem.get("eval_spaces", [])
    per_space = sem.get("per_space", {})
    sizes = sem.get("agg_sizes", [])
    if not spaces or not sizes:
        return html.P(t("no_data", lang), className="text-muted")

    space_col = t("semantic_space", lang)
    model_col = t("model_comparison", lang)

    def sr_row(space_label, mk, sr_dict):
        row = {space_col: space_label, model_col: MODEL_DISPLAY[mk]}
        for n in sizes:
            row[f"SR n={n}"] = f"{sr_dict.get(str(n), 0):.3f}" if sr_dict else "—"
        return row

    rows = []
    for sp in spaces:
        circ = per_space.get(sp, {}).get("bertopic", {}).get("cluster_quality", {}).get("circular")
        sp_label = sp + (f" (⟳ {t('semantic_circular_tag', lang)})" if circ else "")
        for mk in MODEL_KEYS:
            sr = per_space.get(sp, {}).get(mk, {}).get("separation_ratios", {})
            rows.append(sr_row(sp_label if mk == "bertopic" else "", mk, sr))

    # Lexical SR reference block
    lex = sem.get("lexical_separation_ratios", {})
    if lex:
        for i, mk in enumerate(MODEL_KEYS):
            rows.append(sr_row(t("semantic_lexical_sr", lang) if i == 0 else "", mk, lex.get(mk, {})))

    columns = [{"name": c, "id": c} for c in rows[0].keys()]
    return dash_table.DataTable(
        data=rows, columns=columns,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#343a40", "color": "white",
                      "fontWeight": "bold", "textAlign": "center"},
        style_cell={"textAlign": "center", "padding": "8px 12px", "fontSize": "0.85rem"},
        style_data_conditional=[
            {"if": {"filter_query": "{" + model_col + "} = BERTopic"},
             "backgroundColor": "#eef3fb"},
        ],
    )


def _semantic_section(comparison, lang):
    sem = comparison.get("semantic_evaluation", {})
    if not sem or not sem.get("per_space"):
        return None

    return html.Div([
        html.H4(t("semantic_title", lang), className="section-header"),
        html.P(t("semantic_intro", lang), className="mb-3",
               style={"lineHeight": "1.6"}),
        html.Div(
            dcc.Graph(figure=_semantic_ch_chart(sem, lang),
                      config={"displayModeBar": True, "displaylogo": False}),
            className="chart-container",
        ),
        html.H5(t("semantic_cluster_quality_title", lang), className="mt-4"),
        html.Div(_semantic_cluster_quality_table(sem, lang), className="chart-container"),
        html.P(t("semantic_cluster_quality_caption", lang),
               className="text-muted mt-2", style={"fontStyle": "italic"}),
        html.H5(t("semantic_sr_title", lang), className="mt-4"),
        html.Div(_semantic_sr_table(sem, lang), className="chart-container"),
        html.P(t("semantic_sr_caption", lang),
               className="text-muted mt-2", style={"fontStyle": "italic"}),
    ])


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

    # Section 1: Aggregation stabilization curves
    agg_section = html.Div([
        html.H4(t("aggregation_curve", lang), className="section-header"),
        html.Div(
            dcc.Graph(
                figure=_aggregation_curves(comparison, lang),
                config={"displayModeBar": True, "displaylogo": False},
            ),
            className="chart-container",
        ),
    ])

    # Section 2: Aggregation SR table
    sr_section = html.Div([
        html.H4(t("aggregation_table_title", lang), className="section-header"),
        html.Div(
            _aggregation_sr_table(comparison, lang),
            className="chart-container",
        ),
        html.P(
            t("aggregation_explanation", lang),
            className="text-muted mt-3",
            style={"lineHeight": "1.6", "fontStyle": "italic"},
        ),
    ])

    # Section 2b: D3-sem semantic (embedding-space) evaluation
    semantic_section = _semantic_section(comparison, lang)

    # Section 3: Labbe vs JS explanation
    explanation = html.Div([
        html.H4(t("labbe_vs_js_title", lang), className="section-header"),
        html.P(t("labbe_vs_js_text", lang), className="mb-2", style={"lineHeight": "1.6"}),
        html.Div(
            [
                html.Small(
                    "* Distance de Labb\u00e9 : Labbe, D., & Labbe, C. (2001). Inter-textual distance and authorship attribution. "
                    "Journal of Quantitative Linguistics, 8(3), 213-231.",
                    className="text-muted d-block",
                ),
                html.Small(
                    "* Jensen-Shannon : Lin, J. (1991). Divergence measures based on the Shannon entropy. "
                    "IEEE Trans. Information Theory, 37(1), 145-151.",
                    className="text-muted d-block",
                ),
            ],
            className="metric-footnote",
        ),
    ])

    footer = html.Div(t("footer_text", lang), className="footer")

    return html.Div(
        [
            html.H2(t("distances", lang), className="section-header"),
            agg_section,
            sr_section,
            *( [semantic_section] if semantic_section is not None else [] ),
            explanation,
            footer,
        ],
        className="page-container",
    )


# ---------------------------------------------------------------------------
# Page wiring
# ---------------------------------------------------------------------------

layout = html.Div(id="compare-distances-content")


@callback(Output("compare-distances-content", "children"), Input("lang-store", "data"))
def update(lang):
    lang = lang or "fr"
    return build_layout(lang)
