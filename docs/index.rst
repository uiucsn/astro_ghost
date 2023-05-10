Galaxies Hosting Supernovae and other Transients (GHOST)
=========================================================

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Installation Guide

   /source/installation

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   /source/basicusage

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Detailed Tutorials

   /source/detailedtutorials

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Available Modules

   /source/catalogmodules
   /source/preprocessingmodules
   /source/associationmodules
   /source/wrappermodules
   /source/supplementalmodules

.. meta::
   :description lang=en: Automate building, versioning, and hosting of your technical documentation continuously on Read the Docs.

.. Adds a hidden link for the purpose of validating Read the Docs' Mastodon profile
.. raw:: html

   <a style="display: none;" rel="me" href="https://fosstodon.org/@readthedocs">Mastodon</a>

GHOST refers to a database of ~16,175 supernovae and the photometric properties of their host galaxies. Photometry is provided by the Pan-STARRS 3-pi survey in the Northern Hemisphere, 
and SkyMapper in the Southern Hemisphere. GHOST also refers to the software to associate new transients with host galaxies. The package is actively maintained -
if there's a feature you'd like to see, please get in touch!

First time here?
----------------

Check out the following pages: 

.. descriptions here are active

:doc:`/source/installation`
   Install the astro_ghost package.

:doc:`/source/basicusage`
   Retrieve the GHOST database, and associate transients with their most likely host galaxies.

:doc:`For the Experts </source/detailedtutorials>`
   Some more detailed tutorials for those who want to dig into the nuts and bolts of the code. 

If you use the GHOST package or database in your work, please cite the associated paper::

   @ARTICLE{2021ApJ...908..170G,
       author = {{Gagliano}, Alex and {Narayan}, Gautham and {Engel}, Andrew and {Carrasco Kind}, Matias and {LSST Dark Energy Science Collaboration}},
       title = "{GHOST: Using Only Host Galaxy Information to Accurately Associate and Distinguish Supernovae}",
       journal = {\apj},
       year = 2021,
       month = feb,
       volume = {908},
       number = {2},
       eid = {170},
       pages = {170},
       doi = {10.3847/1538-4357/abd02b},
       archivePrefix = {arXiv},
       eprint = {2008.09630},
       primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021ApJ...908..170G},
       adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

