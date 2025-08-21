import os
import sys

# Add the Code folder to sys.path
sys.path.insert(0, os.path.abspath('../Code'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Orbit2Vec'
copyright = '2025, Alexander Theis, Brantley Vose, Dustin Mixon'
author = 'Alexander Theis, Brantley Vose, Dustin Mixon'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Core extension to extract docstrings
    'sphinx.ext.napoleon', # Allows Google/NumPy style docstrings
    'sphinx.ext.todo', 'sphinx.ext.viewcode'

]

# Add after your extensions list
autodoc_mock_imports = [
    'torch',
    'shapely', 
    'shapefile',
    'kagglehub',
    'numpy'  # if used
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
