import os
from setuptools import setup
from Cython.Distutils import build_ext
import numpy as np
from distutils.extension import Extension

with open('requirements.txt') as f:
    required = f.read().splitlines()


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


if __name__ == '__main__':

    setup(
        name="pyegt",
        version="0.0.1",
        py_modules=['pyegt'],
        install_requires=required,
        # Meta information
        author="Julian Garcia",
        author_email="julian.garcia@monash.edu",
        description="PyEGT: Simple python tools for EGT.",
        license="BSD",
        keywords="evolution dynamics complex systems",
        url="http://garciajulian.com",
        classifiers=[
            "Development Status :: 2 - Pre Alpha",
            "Topic :: Utilities",
            "License :: OSI Approved :: BSD License",
        ],
        zip_safe=True,
        include_package_data=True,
        cmdclass={'build_ext': build_ext},
        ext_modules=[
            Extension("pyegt_cython", ["pyegt_cython.pyx"],
                      include_dirs=[np.get_include()])
        ]
    )
