"""Sphinx configuration."""
import datetime
import os
import sys

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath(".."))


def setup(app):
    app.add_css_file("custom.css")


# -- Project information -----------------------------------------------------

project = "glimpse"
author = "Ethan Welty, Douglas Brinkerhoff"
copyright = f"{datetime.datetime.now().year}, {author}"
version = "0.1.0"


# -- Napoleon Extension configuration ------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc configuration -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_default_flags = ["members"]

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
]

doctest_global_setup = """
import glimpse
"""
doctest_test_doctest_blocks = ""

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("http://matplotlib.sourceforge.net/", None),
    "piexif": ("https://piexif.readthedocs.io/en/latest/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
}

autosummary_generate = True
templates_path = ["templates"]

# -- Options for HTML output -------------------------------------------------

html_static_path = ["static"]

# https://sphinx-rtd-theme.readthedocs.io/en/latest/configuring.html
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "style_external_links": False,
    "navigation_depth": 4,
}
