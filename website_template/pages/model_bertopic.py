"""BERTopic model exploration page."""

import dash
from dash import html, Input, Output, callback

from pages._model_common import build_model_layout, register_model_callbacks

dash.register_page(__name__, path="/model-bertopic", name="BERTopic")

layout = html.Div(id="model-bertopic-content")


@callback(Output("model-bertopic-content", "children"), Input("lang-store", "data"))
def update(lang):
    return build_model_layout("bertopic", lang)


register_model_callbacks("bertopic")
