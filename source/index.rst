.. Orbit2Vec documentation master file, created by
   sphinx-quickstart on Tue Aug  5 20:57:53 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Orbit2Vec's Documentation!
======================================

Orbit2Vec is a Python library for orbit-based vector operations, mathematical group transformations, and shapefile processing. This library provides tools for working with geometric data, applying group-theoretic transformations, and converting spatial data into tensor representations.

Features
--------

* **Orbit-based transformations**: Advanced vector operations with distortion tracking
* **Mathematical groups**: Abstract and concrete implementations for filtering operations  
* **Shapefile processing**: Convert geographic data to PyTorch tensors with equidistant sampling
* **Strong typing**: Full type hints for better development experience
* **PyTorch integration**: Native tensor operations for machine learning workflows

Quick Start
-----------

Install the required dependencies and import the main classes:

.. code-block:: python

   import torch
   from orbit2vec import Orbit2Vec
   from group import FromMatrices, Circular
   from shape2matrix import Shape2Matrix

   # Create an orbit-based transformer
   orbit = Orbit2Vec()
   
   # Apply transformations
   vec = torch.randn(5, 1)
   result = orbit.map1(vec)

API Overview
============

The library consists of three main modules:

.. autosummary::
   :toctree: generated/
   :template: module.rst
   
   Code.orbit2vec
   Code.group  
   Code.shape2matrix

Core Classes
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   
   Code.orbit2vec.Orbit2Vec
   Code.group.Group
   Code.group.FromMatrices
   Code.group.Circular
   Code.shape2matrix.Shape2Matrix

Contents
========

.. toctree::
   :maxdepth: 3
   :caption: API Reference:

   orbit2vec
   group
   shape2matrix

.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   
   installation
   examples
   tutorials

.. toctree::
   :maxdepth: 1
   :caption: Development:
   
   contributing
   changelog
   license

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
=======

Copyright 2025, Alexander Theis, Brantley Vose, Dustin Mixon