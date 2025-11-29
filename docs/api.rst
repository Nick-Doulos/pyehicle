API Reference
=============

Preprocessing Module
--------------------

.. automodule:: pyehicle.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Compression
^^^^^^^^^^^

.. autofunction:: pyehicle.preprocessing.spatio_temporal_compress

Map-Matching
^^^^^^^^^^^^

.. autofunction:: pyehicle.preprocessing.leuven
.. autofunction:: pyehicle.preprocessing.meili

Kalman Filtering
^^^^^^^^^^^^^^^^

.. autofunction:: pyehicle.preprocessing.kalman_aeqd_filter

Segmentation
^^^^^^^^^^^^

.. autofunction:: pyehicle.preprocessing.by_time

Interpolation
^^^^^^^^^^^^^

.. autofunction:: pyehicle.preprocessing.by_number_of_points
.. autofunction:: pyehicle.preprocessing.by_sampling_rate

Sampling Rate
^^^^^^^^^^^^^

.. autofunction:: pyehicle.preprocessing.get_sampling_rate

Reconstructing Module
---------------------

.. automodule:: pyehicle.reconstructing
   :members:
   :undoc-members:
   :show-inheritance:

Trajectory Combiner
^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyehicle.reconstructing.trajectory_combiner

Trajectory Refinement
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyehicle.reconstructing.refine_trajectory

Curve Interpolation
^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyehicle.reconstructing.curve_interpolation

Utilities Module
----------------

.. automodule:: pyehicle.utilities
   :members:
   :undoc-members:
   :show-inheritance:

Road Network
^^^^^^^^^^^^

.. autofunction:: pyehicle.utilities.load_road_network

Evaluation
^^^^^^^^^^

.. autofunction:: pyehicle.utilities.f1
.. autofunction:: pyehicle.utilities.precision
.. autofunction:: pyehicle.utilities.recall
.. autofunction:: pyehicle.utilities.rmf

Visualization
^^^^^^^^^^^^^

.. autofunction:: pyehicle.utilities.visualization.single
.. autofunction:: pyehicle.utilities.visualization.multiple
