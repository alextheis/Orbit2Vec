shape2matrix module
===================

The ``shape2matrix`` module provides functionality for converting shapefile data to PyTorch tensor matrices.

.. automodule:: shape2matrix
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

.. autoclass:: shape2matrix.Shape2Matrix
   :members:
   :undoc-members:
   :show-inheritance:

Key Methods
-----------

Shapefile Import
~~~~~~~~~~~~~~~~

.. automethod:: shape2matrix.Shape2Matrix.import_shape

.. automethod:: shape2matrix.Shape2Matrix.set_dataset_path

Data Extraction
~~~~~~~~~~~~~~~

.. automethod:: shape2matrix.Shape2Matrix.extract_shape

.. automethod:: shape2matrix.Shape2Matrix.equidistant

Complete Pipeline
~~~~~~~~~~~~~~~~~

.. automethod:: shape2matrix.Shape2Matrix.process_shapefile

Examples
--------

Complete pipeline usage:

.. code-block:: python

   from shape2matrix import Shape2Matrix

   # Create converter with 100 equidistant points
   converter = Shape2Matrix(100)

   # Process shapefile (complete pipeline)
   tensors = converter.process_shapefile("username/dataset-name")

Step-by-step usage:

.. code-block:: python

   # Step-by-step processing
   converter = Shape2Matrix(50)
   converter.set_dataset_path("username/dataset-name")
   
   # Import shapefile
   reader = converter.import_shape()
   
   # Extract polygon data
   polygons = converter.extract_shape(reader)
   
   # Generate tensors
   tensors = converter.equidistant(polygons)