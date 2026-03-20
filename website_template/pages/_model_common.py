"""
Shared layout builder for model exploration pages (LDA, BERTopic, IRAMUTEQ).

Each model page calls ``build_model_layout(model_key, lang)`` which returns
the full three-tab layout (Topics / Artists / Temporal) and registers all
necessary clientside or server-side callbacks.
"""

from pathlib import Path

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State, dash_table, no_update, ctx

from components.data_loader import get_model
from components.translations import t
from components.colors import TOPIC_COLORS, MODEL_COLORS, topic_color

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Metric references (imported inline to avoid circular deps)
METRIC_REFS = {
    "coherence_cv": {
        "name_fr": "Score de coherence (C_v)",
        "name_en": "Coherence Score (C_v)",
        "citation": "Roder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. WSDM, 399-408.",
        "desc_fr": "La coherence C_v combine l'information mutuelle ponctuelle normalisee (NPMI) avec la similarite cosinus des vecteurs de mots. Des scores plus eleves indiquent des topics plus interpretables. Intervalle : typiquement [0, 1].",
        "desc_en": "C_v coherence combines normalized pointwise mutual information (NPMI) with cosine similarity of word vectors. Higher scores indicate more interpretable topics. Range: typically [0, 1].",
    },
    "silhouette": {
        "name_fr": "Score de silhouette",
        "name_en": "Silhouette Score",
        "citation": "Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. J. Comp. Appl. Math., 20, 53-65.",
        "desc_fr": "Le score de silhouette mesure la similarite d'un point avec son propre cluster par rapport aux autres clusters. Intervalle : [-1, 1], ou 1 = bien clusterise.",
        "desc_en": "The silhouette score measures how similar a point is to its own cluster compared to other clusters. Range: [-1, 1], where 1 = well-clustered.",
    },
    "js_divergence": {
        "name_fr": "Distance de Jensen-Shannon",
        "name_en": "Jensen-Shannon Distance",
        "citation": "Lin, J. (1991). Divergence measures based on the Shannon entropy. IEEE Trans. Inform. Theory, 37(1), 145-151.",
        "desc_fr": "La distance JS est une mesure symetrique de similarite entre distributions de probabilite. Intervalle : [0, 1], ou 0 = distributions identiques.",
        "desc_en": "JS distance is a symmetric measure of similarity between probability distributions. Range: [0, 1], where 0 = identical distributions.",
    },
}


def _metric_explanation(metric_key, lang):
    """Return a static footnote with metric explanation."""
    ref = METRIC_REFS.get(metric_key)
    if not ref:
        return None
    name = ref.get(f"name_{lang}", ref.get("name_en", metric_key))
    desc = ref.get(f"desc_{lang}", ref.get("desc_en", ""))
    citation = ref.get("citation", "")
    return html.Div(
        html.Small(f"* {name} : {desc} ({citation})", className="text-muted"),
        className="metric-footnote",
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _fmt(value, fmt=".3f"):
    if value is None:
        return "N/A"
    try:
        return f"{value:{fmt}}"
    except (TypeError, ValueError):
        return str(value)


def _id(model_key, suffix):
    return f"{model_key}-{suffix}"


# ===================================================================
# TAB 1 -- TOPICS
# ===================================================================

def _topic_dropdown_options(model_data):
    labels = model_data.get("topic_labels", {})
    options = []
    for tid in sorted(labels.keys(), key=lambda x: int(x)):
        options.append({"label": labels[tid], "value": tid})
    return options


def _keyword_bar_chart(model_key, model_data, topic_id, lang):
    topics = model_data.get("topics", {})
    topic = topics.get(str(topic_id), {})
    model_name = model_data.get("name", model_key)

    words, scores = [], []
    score_label = t("weight", lang)

    if model_name == "LDA":
        words = topic.get("words", [])[:20]
        scores = topic.get("probabilities", [])[:20]
        score_label = t("probability", lang)
    elif model_name == "BERTopic":
        ctfidf = topic.get("ctfidf", {})
        words = ctfidf.get("words", [])[:20]
        scores = ctfidf.get("scores", [])[:20]
        score_label = "c-TF-IDF"
    elif model_name == "IRAMUTEQ":
        words = topic.get("words", [])[:20]
        scores = list(range(len(words), 0, -1))
        chi2 = topic.get("chi2", [])
        if chi2 and len(chi2) >= len(words):
            scores = chi2[:20]
            score_label = "Chi2"

    if not words:
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

    words_r = list(reversed(words))
    scores_r = list(reversed(scores))

    color = topic_color(topic_id)
    fig = go.Figure(
        go.Bar(
            x=scores_r, y=words_r, orientation="h",
            marker_color=color,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=t("top_words", lang),
        xaxis_title=score_label,
        margin=dict(l=120, r=20, t=40, b=40),
        plot_bgcolor="white",
        height=max(350, len(words) * 22),
        yaxis=dict(tickfont=dict(size=11)),
        xaxis=dict(gridcolor="#eee"),
    )
    return fig


def _topic_distribution_chart(model_data, selected_topic_id, lang):
    labels = model_data.get("topic_labels", {})
    doc_samples = model_data.get("doc_samples", {})
    topic_evolution = model_data.get("topic_evolution", [])

    counts = {}
    if doc_samples:
        for tid, docs in doc_samples.items():
            counts[tid] = len(docs)
    elif topic_evolution:
        for tid in labels:
            total = sum(row.get(tid, row.get(int(tid), 0)) for row in topic_evolution)
            counts[tid] = total

    sorted_tids = sorted(labels.keys(), key=lambda x: int(x))
    x_labels = [labels.get(tid, f"T{tid}") for tid in sorted_tids]
    y_counts = [counts.get(tid, 0) for tid in sorted_tids]

    colors = []
    for tid in sorted_tids:
        if str(tid) == str(selected_topic_id):
            colors.append(topic_color(tid))
        else:
            colors.append("#d3d3d3")

    fig = go.Figure(
        go.Bar(
            x=x_labels, y=y_counts, marker_color=colors,
            hovertemplate="%{x}<br>%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=t("topic_distribution", lang),
        xaxis_title=t("topic", lang), yaxis_title=t("count", lang),
        margin=dict(l=50, r=20, t=40, b=120),
        plot_bgcolor="white", height=380,
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), gridcolor="#eee"),
        yaxis=dict(gridcolor="#eee"),
    )
    return fig


def _doc_table(model_data, topic_id, lang):
    doc_samples = model_data.get("doc_samples", {})
    docs = doc_samples.get(str(topic_id), [])[:20]

    if not docs:
        return html.P(t("no_data", lang), className="text-muted mt-3")

    rows = []
    for d in docs:
        rows.append({
            t("artist", lang): d.get("artist", ""),
            t("title", lang): d.get("title", ""),
            t("year", lang): d.get("year", ""),
            t("probability", lang): _fmt(d.get("dominant_topic_prob", d.get("prob")), ".3f"),
        })

    columns = [{"name": col, "id": col} for col in rows[0].keys()] if rows else []

    return dash_table.DataTable(
        data=rows, columns=columns, page_size=10, sort_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#343a40", "color": "white",
            "fontWeight": "bold", "textAlign": "center",
        },
        style_cell={
            "textAlign": "center", "padding": "8px 12px",
            "fontSize": "0.85rem",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
        ],
    )


def _keybert_words_section(model_data, topic_id, lang):
    model_name = model_data.get("name", "")
    if model_name != "BERTopic":
        return None

    topic = model_data.get("topics", {}).get(str(topic_id), {})
    keybert = topic.get("keybert", [])
    if not keybert:
        return None

    words_str = ", ".join(keybert[:20]) if isinstance(keybert[0], str) else ", ".join(
        w[0] if isinstance(w, (list, tuple)) else str(w) for w in keybert[:20]
    )
    return html.Div([
        html.H6("KeyBERT keywords", className="mt-3 text-muted"),
        html.P(words_str, className="small"),
    ])


def _embedded_html_section(model_key, model_data, lang="fr"):
    """Return an iframe for pyLDAvis / BERTopic HTML, or static images for IRAMUTEQ."""
    model_name = model_data.get("name", model_key)

    if model_name == "LDA":
        # Check both assets/ and data/ for the file
        assets_path = Path(__file__).resolve().parent.parent / "assets" / "pyldavis.html"
        data_path = DATA_DIR / "pyldavis.html"
        if assets_path.exists() or data_path.exists():
            return html.Div([
                html.H5("pyLDAvis", className="section-header"),
                html.Div(
                    html.Iframe(src="/assets/pyldavis.html"),
                    className="iframe-container",
                ),
            ])
    elif model_name == "BERTopic":
        assets_path = Path(__file__).resolve().parent.parent / "assets" / "interactive_bertopic.html"
        data_path = DATA_DIR / "interactive_bertopic.html"
        if assets_path.exists() or data_path.exists():
            return html.Div([
                html.H5("Interactive BERTopic", className="section-header"),
                html.Div(
                    html.Iframe(src="/assets/interactive_bertopic.html"),
                    className="iframe-container",
                ),
            ])
    elif model_name == "IRAMUTEQ":
        assets_dir = Path(__file__).resolve().parent.parent / "assets"
        dendro = assets_dir / "dendrogramme_iramuteq.png"
        temporal = assets_dir / "chrono_prop_iramuteq.png"
        sections = []
        if dendro.exists():
            sections.append(html.Div([
                html.H5(t("iramuteq_dendrogram", lang), className="section-header"),
                html.Div(
                    html.Img(src="/assets/dendrogramme_iramuteq.png", alt="CHD Dendrogram"),
                    className="iramuteq-img-container chart-container",
                ),
            ]))
        if temporal.exists():
            sections.append(html.Div([
                html.H5(t("iramuteq_temporal_chart", lang), className="section-header"),
                html.Div(
                    html.Img(src="/assets/chrono_prop_iramuteq.png", alt="Temporal Evolution"),
                    className="iramuteq-img-container chart-container",
                ),
            ]))
        if sections:
            return html.Div(sections)
    return None


def _build_topics_tab(model_key, model_data, lang, default_topic_id="0"):
    options = _topic_dropdown_options(model_data)
    if not options:
        return html.P(t("no_data", lang), className="text-muted p-3")

    if default_topic_id not in [o["value"] for o in options]:
        default_topic_id = options[0]["value"]

    keyword_fig = _keyword_bar_chart(model_key, model_data, default_topic_id, lang)
    dist_fig = _topic_distribution_chart(model_data, default_topic_id, lang)
    doc_tbl = _doc_table(model_data, default_topic_id, lang)
    keybert_section = _keybert_words_section(model_data, default_topic_id, lang)
    embedded_section = _embedded_html_section(model_key, model_data, lang)

    children = [
        dbc.Row([
            dbc.Col([
                html.Label(t("select_topic", lang), className="fw-bold mb-1"),
                dcc.Dropdown(
                    id=_id(model_key, "topic-dropdown"),
                    options=options, value=default_topic_id,
                    clearable=False, className="mb-3",
                ),
            ], md=6),
        ]),
        dbc.Row([
            dbc.Col(
                html.Div(
                    dcc.Graph(
                        id=_id(model_key, "keyword-chart"), figure=keyword_fig,
                        config={"displayModeBar": False},
                    ), className="chart-container",
                ), md=6,
            ),
            dbc.Col(
                html.Div(
                    dcc.Graph(
                        id=_id(model_key, "dist-chart"), figure=dist_fig,
                        config={"displayModeBar": False},
                    ), className="chart-container",
                ), md=6,
            ),
        ], className="g-3"),
    ]

    if keybert_section:
        children.append(html.Div(keybert_section, id=_id(model_key, "keybert-section")))
    else:
        children.append(html.Div(id=_id(model_key, "keybert-section")))

    children.append(
        html.Div([
            html.H5(t("example_docs", lang), className="section-header"),
            html.Div(doc_tbl, id=_id(model_key, "doc-table")),
        ])
    )

    if embedded_section:
        children.append(embedded_section)

    # Metric explanation for coherence
    cv_expl = _metric_explanation("coherence_cv", lang)
    if cv_expl:
        children.append(cv_expl)

    return html.Div(children)


# ===================================================================
# TAB 2 -- ARTISTS
# ===================================================================

def _artist_dropdown_options(model_data):
    artist_metrics = model_data.get("artist_metrics", [])
    if not artist_metrics:
        return []
    artists = sorted(set(a.get("artist", "") for a in artist_metrics))
    return [{"label": a, "value": a} for a in artists if a]


def _parse_top_topics(entry, labels):
    """Parse top_3_topics and top_3_ratios (comma-separated strings or lists)."""
    top_topics_raw = entry.get("top_3_topics", entry.get("top_topics", []))
    top_ratios_raw = entry.get("top_3_ratios", entry.get("top_ratios", []))

    # Parse comma-separated strings
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
        # Handle dict items (old format)
        if isinstance(tid, dict):
            t_id = tid.get("topic", tid.get("topic_id", ""))
            pct = tid.get("pct", tid.get("proportion", 0))
            lbl = labels.get(str(t_id), f"T{t_id}")
            pct_str = f"{pct:.1%}" if isinstance(pct, float) else str(pct)
            items.append(html.Li(f"{lbl} ({pct_str})"))
        else:
            lbl = labels.get(str(tid), f"T{tid}")
            if i < len(ratios) and isinstance(ratios[i], (int, float)):
                pct_str = f"{ratios[i]:.1%}"
                items.append(html.Li(f"{lbl} ({pct_str})"))
            else:
                items.append(html.Li(lbl))

    return items


def _artist_profile_card(model_data, artist_name, lang):
    artist_metrics = model_data.get("artist_metrics", [])
    labels = model_data.get("topic_labels", {})

    entry = None
    for am in artist_metrics:
        if am.get("artist") == artist_name:
            entry = am
            break

    if not entry:
        return html.P(t("no_data", lang), className="text-muted")

    dominant = entry.get("dominant_topic", "N/A")
    n_docs = entry.get("n_docs", 0)
    dominant_label = labels.get(str(dominant), f"T{dominant}")

    top_items = _parse_top_topics(entry, labels)

    return dbc.Card([
        dbc.CardHeader(
            html.H5(artist_name, className="mb-0"),
            style={"backgroundColor": "#4a6fa5", "color": "white"},
        ),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.P([
                        html.Strong(t("dominant_topic", lang) + ": "),
                        html.Span(dominant_label),
                    ]),
                    html.P([
                        html.Strong(t("n_docs", lang) + ": "),
                        html.Span(f"{n_docs:,}"),
                    ]),
                ], md=6),
                dbc.Col([
                    html.P(html.Strong(t("top_topics", lang) + ":")),
                    html.Ul(top_items) if top_items else html.P("-", className="text-muted"),
                ], md=6),
            ]),
        ]),
    ], className="shadow-sm")


def _top_artists_table(model_data, filter_topic_id, lang):
    top_artists = model_data.get("top_artists", [])
    labels = model_data.get("topic_labels", {})

    if filter_topic_id is not None and str(filter_topic_id) != "all":
        top_artists = [
            a for a in top_artists
            if str(a.get("topic", "")) == str(filter_topic_id)
        ]

    if not top_artists:
        return html.P(t("no_data", lang), className="text-muted mt-3")

    rows = []
    for a in top_artists[:50]:
        tid = str(a.get("topic", ""))
        rows.append({
            t("topic", lang): labels.get(tid, f"T{tid}"),
            "Rank": a.get("rank", ""),
            t("artist", lang): a.get("artist", ""),
            t("n_docs", lang): a.get("n_docs", ""),
            "%": _fmt(a.get("pct_of_topic", a.get("pct", 0)), ".1%")
                if isinstance(a.get("pct_of_topic", a.get("pct", 0)), float)
                else str(a.get("pct_of_topic", a.get("pct", ""))),
        })

    columns = [{"name": col, "id": col} for col in rows[0].keys()] if rows else []

    return dash_table.DataTable(
        data=rows, columns=columns, page_size=15,
        sort_action="native", filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#343a40", "color": "white",
            "fontWeight": "bold", "textAlign": "center",
        },
        style_cell={
            "textAlign": "center", "padding": "8px 12px",
            "fontSize": "0.85rem",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
        ],
    )


def _build_artists_tab(model_key, model_data, lang):
    artist_options = _artist_dropdown_options(model_data)
    topic_options = _topic_dropdown_options(model_data)
    topic_filter_options = [{"label": "All", "value": "all"}] + topic_options

    default_table = _top_artists_table(model_data, None, lang)
    default_profile = html.P(t("select_artist", lang), className="text-muted")

    return html.Div([
        html.H5(t("artist_explorer", lang), className="section-header"),
        dbc.Row([
            dbc.Col([
                html.Label(t("select_artist", lang), className="fw-bold mb-1"),
                dcc.Dropdown(
                    id=_id(model_key, "artist-dropdown"),
                    options=artist_options, value=None,
                    placeholder=t("search_placeholder", lang),
                    searchable=True, clearable=True, className="mb-3",
                ),
            ], md=6),
        ]),
        html.Div(default_profile, id=_id(model_key, "artist-profile")),
        html.Hr(),
        # Top Artists Table
        html.Div([
            html.H5("Top " + t("artists_tab", lang).lower() + " / " + t("topic", lang).lower(),
                     className="section-header"),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id=_id(model_key, "artist-topic-filter"),
                        options=topic_filter_options, value="all",
                        clearable=False, className="mb-2",
                    ),
                ], md=6),
            ]),
            html.Div(default_table, id=_id(model_key, "top-artists-table")),
        ]),
    ])


# ===================================================================
# TAB 3 -- TEMPORAL
# ===================================================================

def _topic_evolution_area(model_data, lang):
    evolution = model_data.get("topic_evolution", [])
    labels = model_data.get("topic_labels", {})

    if not evolution:
        fig = go.Figure()
        fig.add_annotation(
            text=t("no_data", lang),
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="#999"),
        )
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    years = [row.get("year", row.get("period", "")) for row in evolution]
    topic_ids = sorted(
        [k for k in evolution[0].keys() if k not in ("year", "period")],
        key=lambda x: int(x) if str(x).isdigit() else 0,
    )

    fig = go.Figure()
    for tid in topic_ids:
        vals = [row.get(tid, row.get(str(tid), 0)) for row in evolution]
        color = topic_color(tid)
        label = labels.get(str(tid), f"T{tid}")
        fig.add_trace(go.Scatter(
            x=years, y=vals, mode="lines",
            name=label, stackgroup="one",
            line=dict(width=0.5, color=color),
            fillcolor=color,
            hovertemplate=f"{label}<br>{t('year', lang)}: %{{x}}<br>{t('proportion', lang)}: %{{y:.3f}}<extra></extra>",
        ))

    n_topics = len(topic_ids)
    n_cols = min(5, n_topics)
    fig.update_layout(
        title=t("topic_evolution", lang),
        xaxis_title=t("year", lang),
        yaxis_title=t("proportion", lang),
        margin=dict(l=50, r=20, t=50, b=max(80, 20 + 15 * (n_topics // n_cols))),
        plot_bgcolor="white",
        height=550,
        legend=dict(
            font=dict(size=8),
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            tracegroupgap=2,
        ),
        xaxis=dict(gridcolor="#eee"),
        yaxis=dict(gridcolor="#eee"),
    )
    return fig


def _js_divergence_chart(model_data, lang):
    annual_js = model_data.get("annual_js", [])
    if not annual_js:
        return None

    sample = annual_js[0]
    period_key = "period" if "period" in sample else "year" if "year" in sample else None
    js_key = None
    for k in ("js_divergence", "js_div", "divergence", "mean_js"):
        if k in sample:
            js_key = k
            break

    if not period_key or not js_key:
        return None

    x_vals = [row.get(period_key, "") for row in annual_js]
    y_vals = [row.get(js_key, 0) for row in annual_js]

    fig = go.Figure(
        go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers",
            line=dict(color="#d62728", width=2),
            marker=dict(size=6),
            hovertemplate="%{x}<br>JS: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=t("annual_js", lang),
        xaxis_title=t("year", lang),
        yaxis_title=t("js_divergence", lang),
        margin=dict(l=60, r=20, t=50, b=80),
        plot_bgcolor="white", height=380,
        xaxis=dict(tickangle=-45, gridcolor="#eee"),
        yaxis=dict(gridcolor="#eee"),
    )
    return fig


def _year_topic_heatmap(model_data, lang):
    evolution = model_data.get("topic_evolution", [])
    labels = model_data.get("topic_labels", {})

    if not evolution:
        fig = go.Figure()
        fig.add_annotation(
            text=t("no_data", lang),
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="#999"),
        )
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    years = [str(row.get("year", row.get("period", ""))) for row in evolution]
    topic_ids = sorted(
        [k for k in evolution[0].keys() if k not in ("year", "period")],
        key=lambda x: int(x) if str(x).isdigit() else 0,
    )
    topic_labels_list = [labels.get(str(tid), f"T{tid}") for tid in topic_ids]

    z = []
    for row in evolution:
        z.append([row.get(tid, row.get(str(tid), 0)) for tid in topic_ids])

    fig = go.Figure(
        go.Heatmap(
            z=z, x=topic_labels_list, y=years,
            colorscale="YlOrRd",
            hovertemplate="%{y} | %{x}<br>%{z:.4f}<extra></extra>",
            colorbar=dict(title=t("proportion", lang)),
        )
    )
    fig.update_layout(
        title=t("heatmap", lang) + " (" + t("year", lang) + " x " + t("topic", lang) + ")",
        margin=dict(l=80, r=20, t=50, b=120),
        plot_bgcolor="white",
        height=max(450, len(years) * 18),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _build_temporal_tab(model_key, model_data, lang):
    area_fig = _topic_evolution_area(model_data, lang)
    js_fig = _js_divergence_chart(model_data, lang)
    heatmap_fig = _year_topic_heatmap(model_data, lang)

    children = [
        html.H5(t("temporal_explorer", lang), className="section-header"),
        html.Div(
            dcc.Graph(
                id=_id(model_key, "evolution-area"), figure=area_fig,
                config={"displayModeBar": True, "displaylogo": False},
            ),
            className="chart-container",
        ),
    ]

    if js_fig is not None:
        children.append(
            html.Div(
                dcc.Graph(
                    id=_id(model_key, "js-line"), figure=js_fig,
                    config={"displayModeBar": False},
                ),
                className="chart-container",
            )
        )

    children.append(
        html.Div(
            dcc.Graph(
                id=_id(model_key, "year-heatmap"), figure=heatmap_fig,
                config={"displayModeBar": True, "displaylogo": False},
            ),
            className="chart-container",
        )
    )

    # Metric explanations
    cv_expl = _metric_explanation("js_divergence", lang)
    if cv_expl:
        children.append(cv_expl)

    return html.Div(children)


# ===================================================================
# PUBLIC API -- build_model_layout + register_callbacks
# ===================================================================

def build_model_layout(model_key, lang):
    lang = lang or "fr"
    model_data = get_model(model_key)

    if not model_data:
        return html.Div(
            html.P(t("no_data", lang), className="text-muted p-5 text-center"),
            className="page-container",
        )

    model_name = model_data.get("name", model_key.upper())

    # Metrics summary line
    metrics = model_data.get("metrics", {})
    metric_badges = []
    if "coherence_metrics" in metrics:
        cv = metrics["coherence_metrics"].get("cv")
        if cv is not None:
            metric_badges.append(dbc.Badge(f"C_v = {cv:.3f}", color="info", className="me-2"))
    if "silhouette_metrics" in metrics:
        sil = metrics["silhouette_metrics"].get("silhouette_umap")
        if sil is not None:
            metric_badges.append(dbc.Badge(f"Silhouette = {sil:.3f}", color="info", className="me-2"))
    am = metrics.get("artist_metrics", {})
    mv = am.get("mean_variance") or metrics.get("temporal_metrics", {}).get("mean_variance")
    if mv is not None:
        metric_badges.append(dbc.Badge(f"Mean var. = {mv:.4f}", color="secondary", className="me-2"))

    header = html.Div([
        html.H2(model_name, className="section-header"),
        html.Div(metric_badges, className="mb-3") if metric_badges else None,
    ])

    tabs = dbc.Tabs([
        dbc.Tab(
            _build_topics_tab(model_key, model_data, lang),
            label=t("topics_tab", lang),
            tab_id=_id(model_key, "tab-topics"),
        ),
        dbc.Tab(
            _build_artists_tab(model_key, model_data, lang),
            label=t("artists_tab", lang),
            tab_id=_id(model_key, "tab-artists"),
        ),
        dbc.Tab(
            _build_temporal_tab(model_key, model_data, lang),
            label=t("temporal_tab", lang),
            tab_id=_id(model_key, "tab-temporal"),
        ),
    ], id=_id(model_key, "tabs"), active_tab=_id(model_key, "tab-topics"))

    footer = html.Div(t("footer_text", lang), className="footer")

    return html.Div(
        [header, tabs, footer],
        className="page-container",
    )


# ===================================================================
# CALLBACKS
# ===================================================================

def register_model_callbacks(model_key):

    @callback(
        Output(_id(model_key, "keyword-chart"), "figure"),
        Output(_id(model_key, "dist-chart"), "figure"),
        Output(_id(model_key, "doc-table"), "children"),
        Output(_id(model_key, "keybert-section"), "children"),
        Input(_id(model_key, "topic-dropdown"), "value"),
        Input("lang-store", "data"),
        prevent_initial_call=True,
    )
    def update_topic_view(topic_id, lang):
        lang = lang or "fr"
        model_data = get_model(model_key)
        if not model_data or topic_id is None:
            return no_update, no_update, no_update, no_update

        kw_fig = _keyword_bar_chart(model_key, model_data, topic_id, lang)
        dist_fig = _topic_distribution_chart(model_data, topic_id, lang)
        doc_tbl = _doc_table(model_data, topic_id, lang)
        kb = _keybert_words_section(model_data, topic_id, lang)
        return kw_fig, dist_fig, doc_tbl, kb

    @callback(
        Output(_id(model_key, "artist-profile"), "children"),
        Input(_id(model_key, "artist-dropdown"), "value"),
        Input("lang-store", "data"),
        prevent_initial_call=True,
    )
    def update_artist_profile(artist_name, lang):
        lang = lang or "fr"
        model_data = get_model(model_key)
        if not model_data or not artist_name:
            return html.P(t("select_artist", lang), className="text-muted")
        return _artist_profile_card(model_data, artist_name, lang)

    @callback(
        Output(_id(model_key, "top-artists-table"), "children"),
        Input(_id(model_key, "artist-topic-filter"), "value"),
        Input("lang-store", "data"),
        prevent_initial_call=True,
    )
    def update_top_artists_table(filter_topic, lang):
        lang = lang or "fr"
        model_data = get_model(model_key)
        if not model_data:
            return html.P(t("no_data", lang), className="text-muted")
        return _top_artists_table(model_data, filter_topic, lang)
