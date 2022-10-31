# HOW TO SETUP (FOR PIP)
# Go to the clust directory and remove dist/ and build folders by
# sudo rm -r dist/
# sudo rm -r build/
# Create the bdist_wheel file:
# sudo python3 setup.py sdist bdist_wheel
# Upload to pypi:
# twine upload dist/*
# You will be asked for username (basel) and pword
#
# Updating bioconda:
# Edit this GitHub file:
# https://github.com/bioconda/bioconda-recipes/edit/master/recipes/clust/meta.yaml
# Namely edit the version number and the sha256 number.
#
# Get the sha256 number by this in the command line:
# wget -O- $URL | shasum -a 256
# Where the URL is this (with {{ version }} filled:
# https://pypi.io/packages/source/c/clust/clust-{{ version }}.tar.gz
#
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import sys
from os import path
from clust.scripts.glob import version


def setupmain(args=None):
    if args is None:
        args = sys.argv[1:]

    here = path.abspath(path.dirname(__file__))

    setup(
        name='clust',
        version=version,

        description='Optimised consensus clustering of multiple heterogeneous datasets',
        long_description='Optimised consensus clustering of multiple heterogeneous datasets',

        # The project's main homepage.
        url='https://github.com/baselabujamous/clust',

        # Author details
        author='Basel Abu-Jamous',
        author_email='baselabujamous@gmail.com',

        # Choose your license
        license='OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',

        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Science/Research',
            'Topic :: Software Development :: Build Tools',

            # Pick your license as you wish (should match "license" above)
            'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',

            # Specify the Python versions you support here.
            'Programming Language :: Python :: 3',
        ],

        # What does your project relate to?
        keywords='',

        # You can just specify the packages manually here if your project is
        # simple. Or you can use find_packages().
        packages=find_packages(exclude=['docs', 'tests']),

        # Alternatively, if you want to distribute just a my_module.py, uncomment
        # this:
        #   py_modules=["my_module"],

        # List run-time dependencies here.  These will be installed by pip when
        # your project is installed. For an analysis of "install_requires" vs pip's
        # requirements files see:
        # https://packaging.python.org/en/latest/requirements.html
        install_requires=[
            'joblib>=1.2.0', 
            'portalocker>=2.6.0',
            'numpy>=1.23.4', 
            'scipy>=1.9.3', 
            'pandas>=1.5.0', 
            'matplotlib>=3.6.1', 
            'scikit-learn>=1.1.3',
        ],

        # If there are data files included in your packages that need to be
        # installed, specify them here.  If using Python 2.6 or less, then these
        # have to be included in MANIFEST.in as well.
        # package_data={
        #     'sample': ['package_data.dat'],
        # },

        # Although 'package_data' is the preferred approach, in some case you may
        # need to place data files outside of your packages. See:
        # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
        # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
        # data_files=[('my_data', ['data/data_file'])],

        # To provide executable scripts, use entry points in preference to the
        # "scripts" keyword. Entry points provide cross-platform support and allow
        # pip to create the appropriate form of executable for the target platform.
        entry_points={
            'console_scripts': [
                'clust=clust.__main__:main',
            ],
        },
    )


if __name__ == "__main__":
    setupmain()

