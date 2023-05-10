Installation with Pip
=====================

1. Create a clean conda environment.

2. Run the following code:

.. code-block:: bash

   pip install astro_ghost

Installation with Github
========================
Download `the github repository <https://github.com/uiucsn/astro_ghost>`_ for this project and run

.. code-block:: bash

   python setup.py install

from the main directory.

Testing Installation
=====================
Once installed, run the unit tests with the code below:

.. code-block:: python

   from astro_ghost.unitTests import testAll
   testAll()

If successful, the code will print "Congraulations! Your installation is good to go." (in progress)
