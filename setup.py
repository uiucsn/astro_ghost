import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="astro_ghost",
    version="0.0.46",
    author="Alex Gagliano",
    author_email="gaglian2@illinois.edu",
    description="A package to associate transients with host galaxies, and a database of 16k SNe-host galaxies in PS1.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    package_data={
    '': ['Star_Galaxy_RealisticModel.sav','Star_Galaxy_IdealModel.sav', 'Star_Galaxy_RealisticModel_GHOST_PS1ClassLabels.sav','tonry_ps1_locus.txt'],
    },
    install_requires=['pandas', 'scikit-learn>=0.23.0', 'numpy', 'seaborn', 'matplotlib', 'joypy','astropy>=4.0', 'photutils', 'scipy', 'datetime', 'requests','imblearn','rfpimp','Pillow', 'pyvo', 'astroquery', 'certifi>=2020.06.20'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data = True,
)
