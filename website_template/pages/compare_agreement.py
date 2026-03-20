"""Comparison page: inter-model agreement (ARI / NMI / AMI, Sankey, contingency)."""

import dash
from dash import html, dcc, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from components.data_loader import get_comparison, get_model
from components.translations import t
from components.colors import MODEL_COLORS, TOPIC_COLORS, topic_color

dash.register_page(__name__, path="/compare-agreement", name="Agreement")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAIR_KEYS = [
    "bertopic_vs_lda",
    "bertopic_vs_iramuteq",
    "lda_vs_iramuteq",
]

PAIR_LABELS = {
    "bertopic_vs_lda": "BERTopic vs LDA",
    "bertopic_vs_iramuteq": "BERTopic vs IRAMUTEQ",
    "lda_vs_iramuteq": "LDA vs IRAMUTEQ",
}

PAIR_MODELS = {
    "bertopic_vs_lda": ("bertopic", "lda"),
    "bertopic_vs_iramuteq": ("bertopic", "iramuteq"),
    "lda_vs_iramuteq": ("lda", "iramuteq"),
}

METRIC_REFS = {
    "ari": {
        "desc_fr": "L'indice de Rand ajuste mesure la similarite entre deux clusterings, corrige par le hasard. ARI = (RI - RI_attendu) / (RI_max - RI_attendu). Intervalle : [-1, 1], ou 1 = accord parfait, 0 = aleatoire.",
        "desc_en": "The Adjusted Rand Index measures the similarity between two clusterings, adjusted for chance. ARI = (RI - Expected_RI) / (Max_RI - Expected_RI). Range: [-1, 1], where 1 = perfect agreement, 0 = random.",
        "citation": "Hubert, L., & Arabie, P. (1985). Comparing partitions. Journal of Classification, 2(1), 193-218.",
    },
    "nmi": {
        "desc_fr": "La NMI mesure la dependance mutuelle entre deux clusterings via la theorie de l'information. NMI = 2 x I(X;Y) / (H(X) + H(Y)). Intervalle : [0, 1], ou 1 = clusterings identiques.",
        "desc_en": "NMI measures the mutual dependence between two clusterings using information theory. NMI = 2 x I(X;Y) / (H(X) + H(Y)). Range: [0, 1], where 1 = identical clusterings.",
        "citation": "Strehl, A., & Ghosh, J. (2002). Cluster ensembles. JMLR, 3, 583-617.",
    },
    "ami": {
        "desc_fr": "L'AMI etend la NMI en corrigeant pour l'accord du au hasard. Importante lorsqu'on compare des clusterings avec des nombres de clusters differents. Intervalle : [-1, 1].",
        "desc_en": "AMI extends NMI by adjusting for chance agreement. Important when comparing clusterings with different numbers of clusters. Range: [-1, 1].",
        "citation": "Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic measures for clusterings comparison. JMLR, 11, 2837-2854.",
    },
}


def _fmt(value, fmt=".3f"):
    if value is None:
        return "N/A"
    try:
        return f"{value:{fmt}}"
    except (TypeError, ValueError):
        return str(value)


def _empty_fig(lang):
    fig = go.Figure()
    fig.add_annotation(
        text=t("no_data", lang),
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color="#999"),
    )
    fig.update_layout(
        height=350, margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="white", xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Agreement metrics table
# ---------------------------------------------------------------------------

def _agreement_table(comparison, lang):
    agreement = comparison.get("agreement", {})
    rows = []
    for key in PAIR_KEYS:
        entry = agreement.get(key, {})
        rows.append({
            t("model_comparison", lang): PAIR_LABELS.get(key, key),
            "ARI": _fmt(entry.get("ari")),
            "NMI": _fmt(entry.get("nmi")),
            "AMI": _fmt(entry.get("ami")),
        })

    columns = [
        {"name": t("model_comparison", lang), "id": t("model_comparison", lang)},
        {"name": "ARI", "id": "ARI"},
        {"name": "NMI", "id": "NMI"},
        {"name": "AMI", "id": "AMI"},
    ]

    return dash_table.DataTable(
        data=rows, columns=columns,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#343a40", "color": "white",
            "fontWeight": "bold", "textAlign": "center",
        },
        style_cell={
            "textAlign": "center", "padding": "10px 15px",
            "fontSize": "0.9rem",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
        ],
    )


# ---------------------------------------------------------------------------
# Sankey diagram
# ---------------------------------------------------------------------------

def _build_sankey(pair_key, comparison, lang):
    sankey_data = comparison.get("sankey", {}).get(pair_key, {})
    if not sankey_data:
        return _empty_fig(lang)

    source_model, target_model = PAIR_MODELS.get(pair_key, ("", ""))
    source_topic_labels = get_model(source_model).get("topic_labels", {})
    target_topic_labels = get_model(target_model).get("topic_labels", {})

    raw_source_labels = sankey_data.get("source_labels", [])
    raw_target_labels = sankey_data.get("target_labels", [])

    enriched_source = [source_topic_labels.get(str(lbl), f"T{lbl}") for lbl in raw_source_labels]
    enriched_target = [target_topic_labels.get(str(lbl), f"T{lbl}") for lbl in raw_target_labels]

    all_labels = enriched_source + enriched_target
    n_source = len(enriched_source)

    node_colors = [topic_color(i) for i in range(n_source)] + [topic_color(i) for i in range(len(enriched_target))]

    sources = sankey_data.get("source", [])
    targets = sankey_data.get("target", [])
    values = sankey_data.get("value", [])

    link_colors = []
    for s in sources:
        base = topic_color(s)
        r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
        link_colors.append(f"rgba({r},{g},{b},0.3)")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15, thickness=20,
            line=dict(color="#333", width=0.5),
            label=all_labels, color=node_colors,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=[t_idx + n_source if t_idx < n_source else t_idx for t_idx in targets],
            value=values, color=link_colors,
            hovertemplate="%{source.label} -> %{target.label}<br>" + t("count", lang) + ": %{value:,}<extra></extra>",
        ),
    ))

    source_name = source_model.upper() if source_model != "bertopic" else "BERTopic"
    target_name = target_model.upper() if target_model != "bertopic" else "BERTopic"

    fig.update_layout(
        title=f"{source_name} -> {target_name}",
        margin=dict(l=20, r=20, t=50, b=20),
        height=550, font=dict(size=11),
    )
    return fig


# ---------------------------------------------------------------------------
# Contingency heatmap
# ---------------------------------------------------------------------------

def _build_contingency_heatmap(pair_key, comparison, lang):
    z_data = comparison.get("contingency", {}).get(pair_key)
    labels_data = comparison.get("contingency_labels", {}).get(pair_key, {})

    if z_data is None:
        return _empty_fig(lang)

    row_labels = labels_data.get("rows", [str(i) for i in range(len(z_data))])
    col_labels = labels_data.get("cols", [str(j) for j in range(len(z_data[0]) if z_data else [])])

    source_model, target_model = PAIR_MODELS.get(pair_key, ("", ""))
    source_name = source_model.upper() if source_model != "bertopic" else "BERTopic"
    target_name = target_model.upper() if target_model != "bertopic" else "BERTopic"

    fig = go.Figure(go.Heatmap(
        z=z_data, x=col_labels, y=row_labels,
        colorscale="Viridis",
        hovertemplate=source_name + " T%{y} / " + target_name + " T%{x}<br>" + t("count", lang) + ": %{z:,}<extra></extra>",
        colorbar=dict(title=t("count", lang)),
    ))

    fig.update_layout(
        title=f"{t('contingency_table', lang)} - {PAIR_LABELS.get(pair_key, pair_key)}",
        xaxis_title=target_name + " " + t("topic", lang),
        yaxis_title=source_name + " " + t("topic", lang),
        margin=dict(l=80, r=20, t=50, b=80),
        plot_bgcolor="white", height=500,
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=10)),
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

    # Section 1: Agreement metrics table
    table_section = html.Div([
        html.H4(t("agreement_metrics", lang), className="section-header"),
        html.Div(
            _agreement_table(comparison, lang),
            className="chart-container",
        ),
        # Metric footnotes
        html.Div(
            [
                html.Small(
                    f"* {metric.upper()} : "
                    f"{ref.get(f'desc_{lang}', ref['desc_en'])} "
                    f"({ref['citation']})",
                    className="text-muted d-block",
                )
                for metric, ref in METRIC_REFS.items()
            ],
            className="metric-footnote",
        ),
    ])

    # Section 2: Sankey diagrams (3 tabs)
    sankey_tabs = dbc.Tabs([
        dbc.Tab(
            html.Div(
                dcc.Graph(
                    figure=_build_sankey(pk, comparison, lang),
                    config={"displayModeBar": True, "displaylogo": False},
                ),
                className="chart-container",
            ),
            label=PAIR_LABELS[pk],
            tab_id=f"sankey-tab-{pk}",
        )
        for pk in PAIR_KEYS
    ], id="sankey-tabs", active_tab=f"sankey-tab-{PAIR_KEYS[0]}")

    sankey_section = html.Div([
        html.H4(t("sankey_diagram", lang), className="section-header"),
        sankey_tabs,
    ])

    # Section 3: Contingency heatmaps (3 tabs)
    contingency_tabs = dbc.Tabs([
        dbc.Tab(
            html.Div(
                dcc.Graph(
                    figure=_build_contingency_heatmap(pk, comparison, lang),
                    config={"displayModeBar": True, "displaylogo": False},
                ),
                className="chart-container",
            ),
            label=PAIR_LABELS[pk],
            tab_id=f"cont-tab-{pk}",
        )
        for pk in PAIR_KEYS
    ], id="contingency-tabs", active_tab=f"cont-tab-{PAIR_KEYS[0]}")

    contingency_section = html.Div([
        html.H4(t("contingency_table", lang), className="section-header"),
        contingency_tabs,
    ])

    footer = html.Div(t("footer_text", lang), className="footer")

    return html.Div(
        [
            html.H2(t("agreement", lang), className="section-header"),
            table_section,
            sankey_section,
            contingency_section,
            footer,
        ],
        className="page-container",
    )


# ---------------------------------------------------------------------------
# Page wiring
# ---------------------------------------------------------------------------

layout = html.Div(id="compare-agreement-content")


@callback(Output("compare-agreement-content", "children"), Input("lang-store", "data"))
def update(lang):
    lang = lang or "fr"
    return build_layout(lang)
