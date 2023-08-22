from pathlib import Path
from setuptools import setup, find_packages


install_requires = [
    'bs4>=4.12.2',
    'fastplotlib==0.1.0.a8',
    'ipywidgets==7.7.2',
    'jupyterlab-widgets==1.1.1',
    'pygfx==0.1.10',
    'qtconsole==5.4.0',
    'wgpu==0.8.4'
]

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
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=install_requires,
)