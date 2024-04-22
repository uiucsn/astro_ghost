import os
from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    setup_requires=['setuptools_scm'],
    use_scm_version={
        'write_to': os.path.join('astro_ghost', '_version.py'),
        'write_to_template': "__version__ = '{version}'\n",
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={ 
        'astro_ghost': ['Star_Galaxy_RealisticModel_GHOST_PS1ClassLabels.sav','tonry_ps1_locus.txt','gwgc_good.csv'],
    },
    install_requires=['pandas', 'scikit-learn<1.3.0', 'numpy', 'seaborn', 'matplotlib', 'joypy', 'astropy', 'photutils', 'scipy', 'datetime', 'requests<2.29.0', 'imblearn', 'rfpimp', 'Pillow', 'pyvo', 'astroquery', 'mastcasjobs', 'opencv-python', 'tensorflow', 'sfdmap2', 'importlib_resources'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

