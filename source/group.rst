.. automethod:: orbit2vec.Orbit2Vec.map2

.. automethod:: orbit2vec.Orbit2Vec.map3

.. automethod:: orbit2vec.Orbit2Vec.map4

.. automethod:: orbit2vec.Orbit2Vec.max_inner_product

Examples
--------

Basic usage of the Orbit2Vec class:

.. code-block:: python

   import torch
   from orbit2vec import Orbit2Vec

   # Create an instance
   orbit = Orbit2Vec()

   # Apply map1 transformation
   vec = torch.randn(5, 1)
   result = orbit.map1(vec)

   # Check distortion
   distortion = orbit.get_distortion('map1')
   print(f"Map1 distortion: {distortion}")