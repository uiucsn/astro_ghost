#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from setuptools import setup

version = "0.1.20"

VERSION_TEMPLATE = """
 Note that we need to fall back to the hard-coded version if either
 setuptools_scm can't be imported or setuptools_scm can't determine the
 version, so we catch the generic 'Exception'.
__version__ = '{version}'
""".lstrip()

#setup(
#    use_scm_version={'write_to': os.path.join('astro_ghost', 'version.py'),
#                     'write_to_template': VERSION_TEMPLATE},

#)
#__version__ = vs

import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="astro_ghost",
    version=version,
    author="Alex Gagliano",
    author_email="gaglian2@illinois.edu",
    description="A package to associate transients with host galaxies, and a database of 16k SNe-host galaxies in PS1.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    package_data={
    'astro_ghost': ['Star_Galaxy_RealisticModel.sav','Star_Galaxy_IdealModel.sav', 'Star_Galaxy_RealisticModel_GHOST_PS1ClassLabels.sav','tonry_ps1_locus.txt'],
    },
    install_requires=['pandas', 'sklearn', 'numpy', 'seaborn', 'matplotlib', 'joypy','astropy', 'photutils', 'scipy', 'datetime', 'requests','imblearn','rfpimp','Pillow', 'pyvo', 'astroquery'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
#    use_scm_version={'write_to': os.path.join('astro_ghost', 'version.py'),
#                     'write_to_template': VERSION_TEMPLATE},
)
