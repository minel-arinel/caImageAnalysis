# caImageAnalysis

Two-photon calcium imaging analysis using CaImAn, mesmerize, and fastplotlib

# Installation

1. Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).
2. Open Anaconda Prompt (or Terminal if on Mac).
3. Install [Mamba](https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install).
    - Download the corresponding [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) installer. This will download a .sh bash shell file.
    - Open Anaconda Prompt or Terminal, and change directory to where the downloaded bash shell file is. In your base environment, run 
    
        ```bash
        bash [name of the file.sh]
        ```

4. Create a new environment named `mescore`, and install [mesmerize-core](https://mesmerize-core.readthedocs.io/en/latest/) in it:

    ```bash
    mamba create -n mescore -c conda-forge mesmerize-core
    ```

    This will automatically install [caiman](https://caiman.readthedocs.io/en/master/) too.

5. Activate the environment: 

    ```bash
    mamba activate mescore
    ```

6. Install `caimanmanager`:

    ```bash
    caimanmanager.py install
    ```

7. Run `ipython` and check if `caiman` and `mesmerize_core` are accurately installed:
    - Enter `ipython` in the Anaconda Prompt or Terminal.
    
        ```python
        import caiman
        import mesmerize_core
        print(caiman.__version__)
        print(mesmerize_core.__version__)
        ```
    
8. Install [fastplotlib](https://fastplotlib.readthedocs.io/en/latest/) into the `mescore` environment:

    ```bash
    pip install fastplotlib
    ```

9. Install `git` into the `mescore` environment:

    ```bash
    conda install git
    ```

10. Try installing `simplejpeg`. No worries if you cannot install it, it will just make `fastplotlib` slower:

    ```bash
    pip install simplejpeg
    ```


## Important note

Computing correlation image causes issues in `mesmerize-core`. To overcome the issue, replace your local `mesmerize_core/algorithms/mcorr.py` file with the one under `caImageAnalysis/replace/mesmerize_core/mcorr.py`.

Viewing components causes issues in `caiman`. To overcome the issue, replace your local `caiman/source_extraction/cnmf/estimates.py` file with the one under `caImageAnalysis/replace/caiman/estimates.py`.