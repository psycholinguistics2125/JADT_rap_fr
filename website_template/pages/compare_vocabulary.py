"""Comparison page: vocabulary distinctiveness and chi-squared analysis."""

import dash
from dash import html, dcc, Input, Output, callback, dash_table
import plotly.graph_objects as go

from components.data_loader import get_comparison
from components.translations import t
from components.colors import MODEL_COLORS

dash.register_page(__name__, path="/compare-vocabulary", name="Vocabulary")

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
# Vocabulary distinctiveness bar chart
# ---------------------------------------------------------------------------

def _distinctiveness_chart(comparison, lang):
    distinctiveness = comparison.get("vocabulary", {}).get("distinctiveness", {})

    names, values, colors = [], [], []
    for mk in MODEL_KEYS:
        display = MODEL_DISPLAY[mk]
        v = distinctiveness.get(mk)
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
        title=t("vocab_distinctiveness", lang),
        yaxis_title=t("vocab_distinctiveness", lang),
        margin=dict(l=60, r=20, t=50, b=40),
        plot_bgcolor="white", height=380,
        yaxis=dict(gridcolor="#eee", rangemode="tozero"),
        xaxis=dict(gridcolor="#eee"),
    )
    return fig


# ---------------------------------------------------------------------------
# Chi-squared / n grouped bar chart
# ---------------------------------------------------------------------------

def _chi2_chart(comparison, lang):
    chi2_results = comparison.get("chi2_results", {})
    non_lem = chi2_results.get("non_lemmatized", {})
    lem = chi2_results.get("lemmatized", {})

    if not non_lem and not lem:
        return _empty_fig(lang)

    categories = []
    if non_lem:
        label_non_lem = "Non-lemmatized" if lang == "en" else "Non-lemmatise"
        categories.append(("non_lemmatized", label_non_lem, non_lem))
    if lem:
        label_lem = "Lemmatized" if lang == "en" else "Lemmatise"
        categories.append(("lemmatized", label_lem, lem))

    cat_colors = ["#636EFA", "#EF553B"]
    model_names = [MODEL_DISPLAY[mk] for mk in MODEL_KEYS]

    fig = go.Figure()
    for idx, (key, label, data) in enumerate(categories):
        vals = []
        for mk in MODEL_KEYS:
            entry = data.get(mk, {})
            chi2_n = entry.get("chi2_over_n")
            if chi2_n is None:
                chi2 = entry.get("chi2")
                n = entry.get("n")
                if chi2 is not None and n is not None and n > 0:
                    chi2_n = chi2 / n
            vals.append(chi2_n if chi2_n is not None else 0)

        fig.add_trace(go.Bar(
            name=label, x=model_names, y=vals,
            marker_color=cat_colors[idx % len(cat_colors)],
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            hovertemplate=label + "<br>%{x}: %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        title="Chi2 / n",
        xaxis_title=t("model_comparison", lang), yaxis_title="Chi2 / n",
        barmode="group",
        margin=dict(l=60, r=20, t=50, b=40),
        plot_bgcolor="white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
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

    # Section 1: Vocabulary distinctiveness
    dist_section = html.Div([
        html.H4(t("vocab_distinctiveness", lang), className="section-header"),
        html.Div(
            dcc.Graph(
                figure=_distinctiveness_chart(comparison, lang),
                config={"displayModeBar": False},
            ),
            className="chart-container",
        ),
    ])

    # Section 2: Chi-squared / n
    chi2_section = html.Div([
        html.H4("Chi2 / n", className="section-header"),
        html.Div(
            dcc.Graph(
                figure=_chi2_chart(comparison, lang),
                config={"displayModeBar": False},
            ),
            className="chart-container",
        ),
        # Metric footnote
        html.Div(
            html.Small(
                "* Chi2/n : "
                + (
                    "Chi2/n normalise le chi-deux par la taille de l'echantillon pour permettre la comparaison entre modeles. Plus la valeur est elevee, plus la structuration lexicale par les topics est forte."
                    if lang == "fr" else
                    "Chi2/n normalizes the chi-square by sample size for cross-model comparison. Higher values indicate stronger lexical structuration by topics."
                ),
                className="text-muted",
            ),
            className="metric-footnote",
        ),
    ])

    footer = html.Div(t("footer_text", lang), className="footer")

    return html.Div(
        [
            html.H2(t("vocabulary", lang), className="section-header"),
            dist_section,
            chi2_section,
            footer,
        ],
        className="page-container",
    )


# ---------------------------------------------------------------------------
# Page wiring
# ---------------------------------------------------------------------------

layout = html.Div(id="compare-vocabulary-content")


@callback(Output("compare-vocabulary-content", "children"), Input("lang-store", "data"))
def update(lang):
    lang = lang or "fr"
    return build_layout(lang)
