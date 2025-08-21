import os
import sys

# Add the Code folder to sys.path so Sphinx can import modules
sys.path.insert(0, os.path.abspath('../Code'))

# -- Project information -----------------------------------------------------
project = 'Orbit2Vec'
copyright = '2025, Alexander Theis, Brantley Vose, Dustin Mixon'
author = 'Alexander Theis, Brantley Vose, Dustin Mixon'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # pull in docstrings
    'sphinx.ext.napoleon',      # Google/NumPy docstring support
    'sphinx.ext.viewcode',      # link to highlighted source
    'sphinx.ext.autosummary',   # generate summary tables
]

autosummary_generate = True
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'

# Mock heavy or external dependencies for RTD
autodoc_mock_imports = [
    'torch',
    'shapely',
    'shapefile',
    'kagglehub',
    'numpy',
    'abc'
]

# Ignore unresolved references to external classes
nitpick_ignore = [
    ('py:class', 'torch.Tensor'),
    ('py:class', 'Scalar tensor'),
    ('py:class', 'shapefile.Reader'),
    ('py:class', 'abc.ABC')
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
