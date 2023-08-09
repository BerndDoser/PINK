Parallelized rotation and flipping INvariant Kohonen maps (PINK)
================================================================

|Py-Versions| |PyPi| |LICENCE|

|Build-Status| |ASCL| |Gitter| |Colab|

.. image:: https://github.com/HITS-AIN/PINK/blob/master/doxygen/galaxies_som_hex.jpg
   :width: 400
   :alt: Self-organizing map of radio-synthesis data taken from the Radio Galaxy Zoo project


Requirements
------------

* C++ with ISO 17 standard
* `CMake <https://cmake.org/>`_ >= 3.18
* CUDA >= 9.1 (highly recommended)
* `conan.io <https://conan.io/>`_ (optional for C++ dependencies) or

  * `PyBind11 <https://github.com/pybind/pybind11>`_ (optional for Python interface)
  * `google-test <https://github.com/google/googletest>`_ 1.8.1 (optional for unit tests)

* doxygen 1.8.13 (optional for developer documentation)

Conan.io will install automatically the C++ dependencies (PyBind11 and google-test). Otherwise you can also install these libraries yourself.


Installation
------------

We provide deb- and rpm-packages at https://github.com/HITS-AIN/PINK/releases

or you can install PINK from the sources:

.. code:: sh

   cmake -DCMAKE_INSTALL_PREFIX=<INSTALL_PATH> .
   make install


PyPI installation
-----------------

PINK is also available as `PyPi package <https://pypi.org/project/astro-pink/>`_ which can be installed by

.. code:: sh

   pip install astro-pink


HPC deployment with EasyBuild
-----------------------------

The `EasyBuild <http://easybuilders.github.io/easybuild/>`_ recipe is available at https://github.com/BerndDoser/easybuild-easyconfigs/tree/hits/easybuild/easyconfigs/p/PINK.


Usage
-----

To train a the `self-organizing map <https://en.wikipedia.org/wiki/Self-organizing_map>`_ (SOM) please execute

.. code:: sh

   Pink --train <image-file> <result-file>

where `image-file` is the input file of images for the training and `result-file` is the output file for the trained SOM.
All files are in binary mode described `here <https://github.com/HITS-AIN/PINK/wiki/Description-of-the-binary-file-formats>`_.

To map an image to the trained SOM please execute

.. code:: sh

   Pink --map <image-file> <result-file> <SOM-file>

where `image-file` is the input file of images for the mapping, `SOM-file` is the input file for the trained SOM,
and `result-file` is the output file for the resulting heatmap.

Please use also the command `Pink -h` to get more informations about the usage and the options.


Python scripts
--------------

For conversion and visualization of images and SOM some python scripts are available.

* ``convert_data_binary_file.py``:     Convert binary data file from PINK version 1 to 2
* ``show_heatmap.py``:                Visualize the mapping result
* ``show_images.py``:                 Visualize binary images file format
* ``show_som.py``:                    Visualize binary SOM file format
* ``train.py``:                       SOM training using the PINK Python interface


Publication
-----------

`Kai Lars Polsterer <https://github.com/kai-polsterer>`_, Fabian Gieseke, Christian Igel,
`Bernd Doser <https://github.com/BerndDoser>`_, and
`Nikos Gianniotis <https://github.com/ngiann>`_. Parallelized rotation and flipping INvariant Kohonen maps (PINK) on GPUs.
24th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), pp. 405-410, 2016.
`pdf <https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2016-116.pdf>`_


License
-------

Distributed under the GNU GPLv3 License. See accompanying file LICENSE or copy at http://www.gnu.org/licenses/gpl-3.0.html.


.. |Py-Versions| image:: https://img.shields.io/pypi/pyversions/astro-pink.svg?logo=python
   :target: https://pypi.org/project/astro-pink
.. |Build-Status| image:: https://jenkins.h-its.org/buildStatus/icon?job=AIN/GitHub%20HITS-AIN/PINK/master
   :target: https://jenkins.h-its.org/job/AIN/job/GitHub%20HITS-AIN/job/PINK/job/master/
.. |Gitter| image:: https://badges.gitter.im/HITS-AIN-PINK/Lobby.svg
   :target: https://gitter.im/HITS-AIN-PINK/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |PyPi| image:: https://img.shields.io/pypi/v/astro-pink.svg
   :target: https://github.com/HITS-AIN/PINK/releases
.. |Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/HITS-AIN/PINK/blob/master/colab/pink-train-demo.ipynb
.. |ASCL| image:: https://img.shields.io/badge/ascl-1910.001-blue.svg?colorB=262255
   :target: http://ascl.net/1910.001
.. |LICENCE| image:: https://img.shields.io/badge/license-GPLv3-blue.svg
