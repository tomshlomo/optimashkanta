"""Sphinx configuration."""
project = "OptiMashkanta"
author = "Tom Shlomo"
copyright = "2022, Tom Shlomo"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
