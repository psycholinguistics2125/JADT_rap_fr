import dash
from dash import Dash, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.LITERA],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

# Language store persisted in browser localStorage
app.layout = dbc.Container(
    [
        dcc.Store(id="lang-store", data="fr", storage_type="local"),
        html.Div(id="navbar-container"),
        dash.page_container,
    ],
    fluid=True,
    className="px-0",
)

from components.navbar import create_navbar


@callback(Output("navbar-container", "children"), Input("lang-store", "data"))
def update_navbar(lang):
    return create_navbar(lang)


@callback(
    Output("lang-store", "data"),
    Input("lang-toggle-btn", "n_clicks"),
    Input("lang-store", "data"),
    prevent_initial_call=True,
)
def toggle_language(n_clicks, current_lang):
    if n_clicks:
        return "en" if current_lang == "fr" else "fr"
    return current_lang


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
