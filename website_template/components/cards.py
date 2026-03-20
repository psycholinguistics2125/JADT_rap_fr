import dash_bootstrap_components as dbc
from dash import html


def metric_card(title, value, subtitle=None, color="primary"):
    """Create a metric display card."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="card-subtitle mb-2 text-muted"),
                html.H3(value, className=f"card-title text-{color}"),
                html.P(subtitle, className="card-text small") if subtitle else None,
            ]
        ),
        className="h-100 shadow-sm",
    )


def model_summary_card(model_name, metrics, color, lang="fr"):
    """Create a summary card for a model."""
    items = []

    if "coherence_metrics" in metrics:
        cv = metrics["coherence_metrics"].get("cv")
        if cv:
            items.append(html.Li(f"Coherence C_v: {cv:.3f}"))

    if "silhouette_metrics" in metrics:
        sil = metrics["silhouette_metrics"].get("silhouette_umap")
        if sil:
            items.append(html.Li(f"Silhouette: {sil:.3f}"))

    if "artist_metrics" in metrics:
        am = metrics["artist_metrics"]
        cramers = am.get("cramers_v", "N/A")
        if isinstance(cramers, float):
            items.append(html.Li(f"Cramer's V: {cramers:.3f}"))
        else:
            items.append(html.Li(f"Cramer's V: {cramers}"))

    return dbc.Card(
        [
            dbc.CardHeader(
                html.H5(model_name, className="mb-0"),
                style={"backgroundColor": color, "color": "white"},
            ),
            dbc.CardBody([html.Ul(items, className="list-unstyled mb-0")]),
        ],
        className="h-100 shadow-sm",
    )
