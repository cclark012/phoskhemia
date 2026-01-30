from setuptools import find_packages, setup

from codecs import open
from os import path

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='phoskhemia',
    packages=find_packages(include=['phoskhemia']),
    version='0.1.02',
    description='Library for the handling and analysis of chemical spectroscopic data.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Cole Clark',
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    include_package_data=True,
    install_requires=[
        "numpy", 
        "scipy", 
        "matplotlib", 
        "polars", 
        "numba"
    ],
    setup_requires=[],
    tests_require=[],
)
