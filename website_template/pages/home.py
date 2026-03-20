import random

import dash
from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from components.data_loader import get_corpus, get_model, get_comparison
from components.translations import t
from components.colors import MODEL_COLORS
from components.cards import metric_card, model_summary_card

dash.register_page(__name__, path="/")

# ---------------------------------------------------------------------------
# Abstracts (full text from the JADT 2026 paper)
# ---------------------------------------------------------------------------

ABSTRACT_EN = (
    "A genre in its own right, rap has dominated the French music industry since the 2000s "
    "and now enjoys unprecedented popularity. Caught between legitimization and stigmatization, "
    "French-language rap is becoming increasingly institutionalized, partly through public policy "
    "initiatives (Hammou, 2022). It is also gaining autonomy in economic and media terms, through "
    "growing market share, the emergence of independent labels, specialized journalists, and "
    "dedicated media platforms. Yet while this musical genre was originally rooted in social and "
    "political expression, it may now be undergoing a process of \u201cmainstreaming\u201d "
    "(vari\u00e9tisation), characterized by a thematic diversification driven by commercial "
    "interests (de Courson, 2024). Drawing on a corpus of rap lyrics from the Genius website, "
    "Beno\u00eet de Courson conducted lexicometric and topic modeling analyses on over 37,000 "
    "French-language rap tracks released between 1992 and 2024. He identified seven topics and "
    "opened up several avenues of research that we aim to extend. First, the corpus is "
    "particularly complex, featuring extensive use of slang, verlan (French backslang), and "
    "various forms of textual noise, which call for dedicated linguistic preprocessing. Second, "
    "the seven identified topics may obscure certain thematic nuances, as the author himself "
    "acknowledges."
)

ABSTRACT_EN_2 = (
    "Our proposal involves conducting a systematic comparative analysis of this same corpus "
    "using three complementary approaches: Descending Hierarchical Classification (DHC) via "
    "Iramuteq (Ratinaud, 2014), probabilistic topic modeling with LDA, and neural topic "
    "modeling with BERTopic. The comparison draws on multiple dimensions, including inter-model "
    "agreement, thematic distribution across artists, temporal dynamics, vocabulary overlap, and "
    "intra-topic coherence. Through this work, we aim not only to identify finer-grained themes "
    "and better characterize the evolution of French-language rap but also to contribute "
    "methodologically to the growing literature on comparing topic modeling approaches."
)

ABSTRACT_FR = (
    "Genre \u00e0 part, le rap submerge l\u2019industrie musicale fran\u00e7aise depuis les "
    "ann\u00e9es 2000 et conna\u00eet d\u00e9sormais une notori\u00e9t\u00e9 sans "
    "pr\u00e9c\u00e9dent. Entre l\u00e9gitimation et stigmatisation, le rap francophone "
    "s\u2019institutionnalise, notamment \u00e0 travers l\u2019action des pouvoirs publics "
    "(Hammou, 2022). Il tend \u00e0 s\u2019autonomiser sur le plan \u00e9conomique et "
    "m\u00e9diatique (en parts de march\u00e9, par la cr\u00e9ation de labels "
    "ind\u00e9pendants, de journalistes sp\u00e9cialis\u00e9s et de m\u00e9dias "
    "d\u00e9di\u00e9s). Mais si ce genre musical est initialement empreint de revendications "
    "sociales et politiques, il serait d\u00e9sormais sujet \u00e0 un ph\u00e9nom\u00e8ne de "
    "vari\u00e9tisation, soit une diversification des th\u00e9matiques \u00e0 vocation "
    "commerciale (de Courson, 2024). Gr\u00e2ce \u00e0 la constitution d\u2019un corpus de "
    "textes de rap issus du site Genius, Beno\u00eet de Courson propose des analyses "
    "lexicom\u00e9triques et de topic modeling sur plus de 37 000 morceaux de rap francophone "
    "parus entre 1992 et 2024. Il identifie ainsi 7 topics et ouvre plusieurs pistes de "
    "recherche que nous souhaitons prolonger. En premier lieu, il s\u2019agit d\u2019un corpus "
    "complexe, largement constitu\u00e9 de mots d\u2019argot, de verlan et de bruits divers, "
    "ce qui motive un traitement linguistique d\u00e9di\u00e9. Ensuite, les 7 topics "
    "identifi\u00e9s masquent potentiellement certaines sp\u00e9cificit\u00e9s th\u00e9matiques, "
    "comme le souligne d\u2019ailleurs l\u2019auteur."
)

ABSTRACT_FR_2 = (
    "Notre proposition consiste \u00e0 mener une analyse comparative syst\u00e9matique de ce "
    "m\u00eame corpus selon trois approches compl\u00e9mentaires : une Classification "
    "Hi\u00e9rarchique Descendante (CHD) via Iramuteq (Ratinaud, 2014), un topic modeling "
    "probabiliste par LDA et un topic modeling neuronal par BERTopic. La comparaison s\u2019appuie "
    "sur plusieurs dimensions, notamment l\u2019accord inter-mod\u00e8les, la distribution "
    "th\u00e9matique selon les artistes, les dynamiques temporelles, le recouvrement lexical et "
    "la coh\u00e9rence intra-topic. \u00c0 travers ce travail, nous souhaitons non seulement "
    "identifier des th\u00e9matiques plus fines et mieux caract\u00e9riser l\u2019\u00e9volution "
    "du rap francophone, mais \u00e9galement contribuer m\u00e9thodologiquement \u00e0 la "
    "litt\u00e9rature croissante portant sur la comparaison des approches de topic modeling."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(value, fmt=".3f"):
    if value is None:
        return "N/A"
    try:
        return f"{value:{fmt}}"
    except (TypeError, ValueError):
        return str(value)


def _build_year_chart(corpus, lang):
    year_dist = corpus.get("year_distribution", {})
    years = sorted(year_dist.keys())
    counts = [year_dist[y] for y in years]

    fig = go.Figure(
        go.Bar(
            x=years,
            y=counts,
            marker_color="#4a6fa5",
            hovertemplate="%{x}: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title=t("year", lang),
        yaxis_title=t("count", lang),
        margin=dict(l=50, r=20, t=20, b=50),
        plot_bgcolor="white",
        height=300,
        xaxis=dict(tickmode="linear", dtick=2, gridcolor="#eee"),
        yaxis=dict(gridcolor="#eee"),
    )
    return fig


def _build_agreement_table(comparison, lang):
    agreement = comparison.get("agreement", {})
    pair_labels = {
        "bertopic_vs_lda": "BERTopic vs LDA",
        "bertopic_vs_iramuteq": "BERTopic vs IRAMUTEQ",
        "lda_vs_iramuteq": "LDA vs IRAMUTEQ",
    }
    rows = []
    for key, label in pair_labels.items():
        entry = agreement.get(key, {})
        rows.append({
            "pair": label,
            "ARI": _fmt(entry.get("ari")),
            "NMI": _fmt(entry.get("nmi")),
            "AMI": _fmt(entry.get("ami")),
        })

    columns = [
        {"name": t("model_comparison", lang), "id": "pair"},
        {"name": "ARI", "id": "ARI"},
        {"name": "NMI", "id": "NMI"},
        {"name": "AMI", "id": "AMI"},
    ]

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
            "padding": "10px 15px",
            "fontSize": "0.9rem",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
        ],
    )


def _get_cramers_v(model_key, comparison):
    artist_sep = comparison.get("artist_separation", {})
    entry = artist_sep.get(model_key, {})
    return entry.get("cramers_v")


def _corpus_sample_table(samples, lang, page_size=10):
    """Build a table of corpus sample documents."""
    if not samples:
        return html.P(t("no_data", lang), className="text-muted")

    rows = []
    for d in samples:
        row = {
            t("artist", lang): d.get("artist", ""),
            t("title", lang): d.get("title", ""),
            t("year", lang): d.get("year", ""),
        }
        if "excerpt" in d:
            row[t("lyrics_excerpt", lang)] = d.get("excerpt", "")
        rows.append(row)

    if not rows:
        return html.P(t("no_data", lang), className="text-muted")

    columns = [{"name": col, "id": col} for col in rows[0].keys()]

    return dash_table.DataTable(
        data=rows,
        columns=columns,
        page_size=page_size,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#343a40",
            "color": "white",
            "fontWeight": "bold",
            "textAlign": "center",
        },
        style_cell={
            "textAlign": "left",
            "padding": "8px 12px",
            "fontSize": "0.85rem",
            "maxWidth": "400px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_cell_conditional=[
            {"if": {"column_id": t("lyrics_excerpt", lang)}, "whiteSpace": "normal"},
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
        ],
    )


# ---------------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------------

def build_layout(lang, shuffle_seed=None):
    corpus = get_corpus()
    bertopic = get_model("bertopic")
    lda = get_model("lda")
    iramuteq = get_model("iramuteq")
    comparison = get_comparison()

    year_range = corpus.get("year_range", [])
    period_str = f"{year_range[0]} - {year_range[1]}" if len(year_range) == 2 else "N/A"

    # ---- Header: Title + Authors (centered, simple) ----
    header = html.Div(
        [
            html.H3(
                t("pub_title", lang),
                className="mb-2",
                style={"color": "#2c3e50", "lineHeight": "1.5", "fontWeight": "600"},
            ),
            html.P(
                t("pub_authors", lang),
                className="text-muted mb-0",
                style={"fontSize": "0.95rem"},
            ),
        ],
        className="home-header",
    )

    # ---- Abstract (full, static) ----
    if lang == "fr":
        abstract_paragraphs = [
            html.P(ABSTRACT_FR, className="mb-2"),
            html.P(ABSTRACT_FR_2),
        ]
    else:
        abstract_paragraphs = [
            html.P(ABSTRACT_EN, className="mb-2"),
            html.P(ABSTRACT_EN_2),
        ]

    abstract_section = html.Div(
        [
            html.H5(t("abstract_title", lang), className="section-header"),
            html.Div(abstract_paragraphs, className="home-abstract"),
        ]
    )

    # ---- Goal of this website ----
    goal_section = html.Div(
        html.P(
            t("website_goal", lang),
            style={"fontSize": "0.95rem", "lineHeight": "1.6", "fontStyle": "italic"},
            className="text-muted",
        ),
        className="home-abstract mt-3 mb-2",
    )

    separator = html.Hr(style={"marginTop": "1.5rem", "marginBottom": "1.5rem"})

    # ---- Corpus Summary Cards ----
    corpus_section = html.Div(
        [
            html.H4(t("corpus_summary", lang), className="section-header"),
            dbc.Row(
                [
                    dbc.Col(metric_card(
                        t("documents", lang).capitalize(),
                        f"{corpus.get('n_documents', 0):,}",
                    ), md=4),
                    dbc.Col(metric_card(
                        t("unique_artists", lang).capitalize(),
                        f"{corpus.get('n_artists', 0):,}",
                    ), md=4),
                    dbc.Col(metric_card(
                        t("year_range", lang).capitalize(),
                        period_str,
                    ), md=4),
                ],
                className="g-3",
            ),
        ]
    )

    # ---- Year Distribution Chart ----
    year_chart_section = html.Div(
        [
            html.H4(t("year_distribution", lang), className="section-header"),
            html.Div(
                dcc.Graph(
                    figure=_build_year_chart(corpus, lang),
                    config={"displayModeBar": False},
                ),
                className="chart-container",
            ),
        ]
    )

    # ---- Corpus Explorer ----
    corpus_explorer = html.Div(
        [
            html.H4(t("explore_corpus", lang), className="section-header"),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id="corpus-artist-filter",
                        options=[],
                        value=None,
                        placeholder=t("filter_by_artist", lang),
                        searchable=True,
                        clearable=True,
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Button(
                        t("shuffle", lang),
                        id="corpus-shuffle-btn",
                        color="secondary",
                        size="sm",
                        className="mt-1",
                    ),
                ], md=2),
            ], className="mb-3"),
            html.Small(
                t("corpus_sample_note", lang),
                className="text-muted d-block mb-2",
            ),
            html.Div(
                _corpus_sample_table(
                    corpus.get("doc_samples", [])[:10] if shuffle_seed is None
                    else random.Random(shuffle_seed).sample(
                        corpus.get("doc_samples", []),
                        min(10, len(corpus.get("doc_samples", []))),
                    ),
                    lang,
                ),
                id="corpus-sample-table",
            ),
        ]
    )

    # ---- Model Summary Cards ----
    model_configs = [
        ("BERTopic", "bertopic", bertopic),
        ("LDA", "lda", lda),
        ("IRAMUTEQ", "iramuteq", iramuteq),
    ]

    model_cards = []
    for display_name, key, model_data in model_configs:
        metrics = dict(model_data.get("metrics", {}))
        if "artist_metrics" not in metrics:
            metrics["artist_metrics"] = {}
        if metrics["artist_metrics"].get("cramers_v") is None:
            cv = _get_cramers_v(key, comparison)
            if cv is not None:
                metrics["artist_metrics"]["cramers_v"] = cv

        model_cards.append(
            dbc.Col(
                model_summary_card(
                    display_name,
                    metrics,
                    MODEL_COLORS.get(display_name, "#6c757d"),
                    lang=lang,
                ),
                md=4,
            )
        )

    model_section = html.Div(
        [
            html.H4(t("model_comparison", lang), className="section-header"),
            dbc.Row(model_cards, className="g-3"),
        ]
    )

    # ---- Agreement Overview Table ----
    agreement_section = html.Div(
        [
            html.H4(t("agreement_metrics", lang), className="section-header"),
            html.Div(
                _build_agreement_table(comparison, lang),
                className="chart-container",
            ),
        ]
    )

    footer = html.Div(t("footer_text", lang), className="footer")

    return html.Div(
        [
            header,
            abstract_section,
            goal_section,
            separator,
            corpus_section,
            year_chart_section,
            corpus_explorer,
            model_section,
            agreement_section,
            footer,
        ],
        className="page-container",
    )


# ---------------------------------------------------------------------------
# Page layout + callbacks
# ---------------------------------------------------------------------------

layout = html.Div(id="home-content")


@callback(Output("home-content", "children"), Input("lang-store", "data"))
def render_home(lang):
    lang = lang or "fr"
    return build_layout(lang)


@callback(
    Output("corpus-sample-table", "children"),
    Input("corpus-shuffle-btn", "n_clicks"),
    Input("corpus-artist-filter", "value"),
    State("lang-store", "data"),
    prevent_initial_call=True,
)
def shuffle_corpus(n_clicks, artist_filter, lang):
    lang = lang or "fr"
    corpus = get_corpus()

    if artist_filter:
        # Use all_docs for full artist song list
        all_docs = corpus.get("all_docs", [])
        filtered = [d for d in all_docs if d.get("artist") == artist_filter]
        return _corpus_sample_table(filtered, lang, page_size=20)

    # Shuffle mode: use random sample from doc_samples
    samples = corpus.get("doc_samples", [])
    seed = n_clicks if n_clicks else 0
    rng = random.Random(seed)
    display = rng.sample(samples, min(10, len(samples)))
    return _corpus_sample_table(display, lang)


@callback(
    Output("corpus-artist-filter", "options"),
    Input("lang-store", "data"),
)
def populate_artist_filter(lang):
    corpus = get_corpus()
    # Use all_docs for the full artist list
    all_docs = corpus.get("all_docs", [])
    if all_docs:
        artists = sorted(set(d.get("artist", "") for d in all_docs if d.get("artist")))
    else:
        samples = corpus.get("doc_samples", [])
        artists = sorted(set(s.get("artist", "") for s in samples if s.get("artist")))
    return [{"label": a, "value": a} for a in artists]
