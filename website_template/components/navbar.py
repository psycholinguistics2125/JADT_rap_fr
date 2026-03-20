import dash_bootstrap_components as dbc
from dash import html

from components.translations import t


def create_navbar(lang="fr"):
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(t("site_title", lang), href="/", className="fw-bold"),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink(t("home", lang), href="/")),
                            dbc.DropdownMenu(
                                [
                                    dbc.DropdownMenuItem(
                                        "BERTopic", href="/model-bertopic"
                                    ),
                                    dbc.DropdownMenuItem("LDA", href="/model-lda"),
                                    dbc.DropdownMenuItem(
                                        "IRAMUTEQ", href="/model-iramuteq"
                                    ),
                                ],
                                label=t("models", lang),
                                nav=True,
                            ),
                            dbc.DropdownMenu(
                                [
                                    dbc.DropdownMenuItem(
                                        t("agreement", lang),
                                        href="/compare-agreement",
                                    ),
                                    dbc.DropdownMenuItem(
                                        t("artists", lang), href="/compare-artists"
                                    ),
                                    dbc.DropdownMenuItem(
                                        t("temporal", lang), href="/compare-temporal"
                                    ),
                                    dbc.DropdownMenuItem(
                                        t("vocabulary", lang),
                                        href="/compare-vocabulary",
                                    ),
                                    dbc.DropdownMenuItem(
                                        t("distances", lang),
                                        href="/compare-distances",
                                    ),
                                ],
                                label=t("comparison", lang),
                                nav=True,
                            ),
                            dbc.NavItem(
                                dbc.Button(
                                    t("lang_toggle", lang),
                                    id="lang-toggle-btn",
                                    color="outline-light",
                                    size="sm",
                                    className="ms-2",
                                )
                            ),
                        ],
                        className="ms-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ],
            fluid=True,
        ),
        color="primary",
        dark=True,
        sticky="top",
        className="shadow-sm",
    )
