"""IRAMUTEQ model exploration page."""

import dash
from dash import html, Input, Output, callback

from pages._model_common import build_model_layout, register_model_callbacks

dash.register_page(__name__, path="/model-iramuteq", name="IRAMUTEQ")

layout = html.Div(id="model-iramuteq-content")


@callback(Output("model-iramuteq-content", "children"), Input("lang-store", "data"))
def update(lang):
    return build_model_layout("iramuteq", lang)


register_model_callbacks("iramuteq")
