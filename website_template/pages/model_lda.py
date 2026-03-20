"""LDA model exploration page."""

import dash
from dash import html, Input, Output, callback

from pages._model_common import build_model_layout, register_model_callbacks

dash.register_page(__name__, path="/model-lda", name="LDA")

layout = html.Div(id="model-lda-content")


@callback(Output("model-lda-content", "children"), Input("lang-store", "data"))
def update(lang):
    return build_model_layout("lda", lang)


register_model_callbacks("lda")
