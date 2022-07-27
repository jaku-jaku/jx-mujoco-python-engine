#!/usr/bin/env python3
import os
from setuptools import find_packages, setup

VERSION = '0.0.1'

INSTALL_REQUIRES = (
    [
        'mujoco >= 2.1.5',
    ]
)

setup(
    name='mujoco-python-engine',
    version=VERSION,
    author='Jack Xu',
    author_email='projectbyjx@gmail.com',
    url='https://github.com/jaku-jaku/jx-mujoco-python-engine',
    description='Mujoco Engine in Python',
    long_description='',
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
    ],
    zip_safe=False,
)
