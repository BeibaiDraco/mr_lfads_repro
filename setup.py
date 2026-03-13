from setuptools import find_packages, setup

setup(
    name='mr-lfads-repro',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.23',
        'scipy>=1.10',
        'torch>=2.0',
        'tqdm>=4.64',
    ],
)
