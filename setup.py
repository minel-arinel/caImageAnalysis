from pathlib import Path
from setuptools import setup, find_packages


with open(Path(__file__).parent.joinpath("README.md")) as f:
    readme = f.read()

setup(
    name='caImageAnalysis',
    version=1.0,
    description='Two-photon calcium imaging analysis',
    license='MIT',
    long_description=readme,
    author='Minel Arinel',
    author_email='minelarinel@gmail.com', 
    url='https://github.com/minel-arinel/caImageAnalysis',
    packages=find_packages()
)