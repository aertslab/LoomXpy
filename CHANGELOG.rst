=========
Changelog
=========

0.4.0 (2021-07-15)
------------------

* ``export`` function does now export cluster markers Closes `#8`_.
* Default embedding (i.e.: ``ra.Embedding``) if present in .loom file is now read and stored in LoomX object `#182c0d1`_.

.. _#8: https://github.com/aertslab/LoomXpy/issues/8
.. _#182c0d1: https://github.com/aertslab/LoomXpy/commit/182c0d15f0a6a2bcaf1d264951260c67839f2d93

0.3.1 (2021-07-14)
------------------

* Unable to read loom when markers not present along with clustering Fixes `#7`_.

.. _#7: https://github.com/aertslab/LoomXpy/issues/7

0.3.0 (2021-06-18)
------------------

* Read LoomX files and store data (annotations, metrics, clusterings, cluster markers and embeddings) into LoomX object `#3`_.

.. _#3: https://github.com/aertslab/LoomXpy/issues/3

0.2.0 (2021-04-20)
------------------

* Initial abstraction draft and basic export to SCope.


0.1.0 (2021-03-03)
------------------

* First release with SCopeLoom class from VSN.
