Getting Started
===============

`orbit2vec` is a **PyTorch library for constructing bi-Lipschitz invariant maps** 
that embed quotient spaces ``V / G`` into Hilbert space. It supports **group-invariant operations** 
for orbit recovery, symmetry-aware learning, and invariant feature extraction.  

The library integrates three main components:

1. `shape2matrix` – converts geometrical shapes into equidistant matrix representations.  
2. `orbit2vec` – provides invariant maps for vectors and matrices.  
3. `group` – defines abstract groups and specific group actions for invariance.  

Features
--------

- **Shape-to-Matrix Conversion**: Extract polygon exteriors, interpolate equidistant points, and create PyTorch matrices.  
- **PCA Visualization**: Visualize shape matrices and vector embeddings in 2D PCA space.  
- **Invariant Maps**: ``map1``–``map4`` for vectors/matrices with distortion guarantees.  
- **Sorting-Based Invariance**: ``map2`` produces order-invariant embeddings for scalar lists.  
- **Centroid Normalization**: ``map4`` subtracts column means before embedding.  
- **Matrix Square Roots**: Compute stable symmetric square roots for embeddings.  
- **Group Actions**: Define abstract groups, groups from matrices, and circular groups.  
- **Max Inner Product**: Compute max inner product over a group of isometries or circular shifts.  
- **Distortion Tracking**: Store, retrieve, and clear map distortions.  

Installation
------------

.. code-block:: bash

    pip install orbit2vec

Dependencies:

- ``torch``  
- ``shapefile``, ``shapely``, ``kagglehub`` (for ``shape2matrix``)  
- ``scikit-learn``  
- ``matplotlib``  
- ``numpy``  

Usage Example
-------------

Shape-to-Matrix
^^^^^^^^^^^^^^^

.. code-block:: python

    from shape2matrix import shape2matrix

    # Load shapes from a shapefile dataset
    s2m = shape2matrix(num=50)
    reader = s2m.import_shape()
    polygons = s2m.extract_shape(reader)
    matrices = s2m.equidistant(polygons)

Orbit2Vec Maps
^^^^^^^^^^^^^^

.. code-block:: python

    from orbit2vec import orbit2vec
    import torch

    orb = orbit2vec()
    v = torch.randn(5, 1)
    embedded = orb.map1(v)         # map1 embedding
    M = torch.randn(5, 5)
    processed = orb.map4(M)        # map4 embedding

Group-Invariant Inner Product
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from group import from_matrices, circular

    # Define a group from matrices
    G = from_matrices([torch.eye(5), -torch.eye(5)])
    max_ip_fn = G.max_filter(template=M)
    max_ip = max_ip_fn(torch.randn(5))

    # Circular group actions
    C = circular()
    f, g = torch.randn(10), torch.randn(10)
    max_circ = C.max_filter()(f, g)

Classes & Methods
-----------------

+-------------------------------+--------------------------------------------------------------+
| Class / Method                | Description                                                  |
+===============================+==============================================================+
| ``shape2matrix``              | Converts shapes to equidistant PyTorch matrices.            |
+-------------------------------+--------------------------------------------------------------+
| ``shape2matrix.import_shape()``| Loads a shapefile dataset.                                   |
+-------------------------------+--------------------------------------------------------------+
| ``shape2matrix.extract_shape(reader)`` | Extracts polygon coordinates.                          |
+-------------------------------+--------------------------------------------------------------+
| ``shape2matrix.equidistant(polygons)`` | Creates equidistant point matrices.                     |
+-------------------------------+--------------------------------------------------------------+
| ``shape2matrix.pca(data)``    | Visualizes embeddings in 2D PCA space.                       |
+-------------------------------+--------------------------------------------------------------+
| ``orbit2vec.map1(vec)``       | Gramian-based vector embedding.                               |
+-------------------------------+--------------------------------------------------------------+
| ``orbit2vec.map2(list)``      | Sort-based invariant embedding.                               |
+-------------------------------+--------------------------------------------------------------+
| ``orbit2vec.map3(matrix)``    | Matrix square root embedding.                                 |
+-------------------------------+--------------------------------------------------------------+
| ``orbit2vec.map4(matrix)``    | Column-mean normalized matrix embedding.                     |
+-------------------------------+--------------------------------------------------------------+
| ``orbit2vec.max_inner_product(x, y, group)`` | Max inner product over group elements.            |
+-------------------------------+--------------------------------------------------------------+
| ``orbit2vec.get_distortion(name)`` | Returns map distortion.                                  |
+-------------------------------+--------------------------------------------------------------+
| ``group.Group``               | Abstract base class for group actions.                       |
+-------------------------------+--------------------------------------------------------------+
| ``group.from_matrices``       | Max inner product for groups defined by matrices.            |
+-------------------------------+--------------------------------------------------------------+
| ``group.circular``            | Circular group actions, Fourier-based invariance.            |
+-------------------------------+--------------------------------------------------------------+

Applications
------------

- Symmetry-aware machine learning  
- Orbit recovery in geometric spaces  
- Group-invariant feature extraction  
- Bi-Lipschitz embedding of structured data  
- Shape analysis and PCA visualization  
