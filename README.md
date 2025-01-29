# caImageAnalysis

Two-photon calcium imaging analysis using CaImAn, mesmerize, and fastplotlib.

# Installation
To set up Mamba, follow these steps:

1. Install Anaconda by following the instructions on the [official website](https://www.anaconda.com/download/success).

Enter the following commands into the Terminal on Mac.

2. Install [Homebrew](https://brew.sh):

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

3. Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html):

    ```bash
    brew install micromamba
    ```

4. If you get a `zsh: command not found: brew` error, run the following commands in this order and then try the above command again:

    ```bash
    cd /opt/homebrew/bin/

    PATH=$PATH:/opt/homebrew/bin

    echo export PATH=$PATH:/opt/homebrew/bin >> ~/.zshrc
    ```

5. After `micromamba` is successfully installed, restart your terminal.

6. Verify the installation:

    ```bash
    micromamba --version
    ```

7. Modify the shell configuration and set the root prefix location to activate environments more easily:

    ```bash
    micromamba shell init --shell zsh --root-prefix=~/.local/share/mamba
    ```

8. For analysis of two-photon recordings of neurons, clone the following repository:

    ```bash
    git clone https://github.com/minel-arinel/caImageAnalysis.git
    ```

9. Create the `mescore` environment:

    ```bash
    cd caImageAnalysis

    micromamba create -f environment.yml
    ```

10. Activate the `mescore` environment:

    ```bash
    micromamba activate mescore
    ```

11. Confirm that the correct version of Python is installed (3.10.12):

    ```bash
    python --version
    ```

12. Confirm that `caiman` and `mesmerize-core` are installed successfully:

    ```bash
    ipython
    ```

    ```python
    # Run in ipython
    import caiman
    import mesmerize_core
    print(caiman.__version__)  # should be 1.9.15
    print(mesmerize_core.__version__)  # should be 0.2.2
    ```

13. Finally, there are some bugs in these specific versions of `caiman` and `mesmerize-core`. They are fixed in later versions, but these versions also make the code incompatible. Therefore, we will manually fix this problem by replacing some of the `.py` files in these packages with the ones in the `caImageAnalysis` repository. 

Replace the `~/.local/share/mamba/envs/mescore/lib/python3.10/site-packages/caiman/source_extraction/cnmf/estimates.py` file with the `~/caImageAnalysis/caImageAnalysis/replace/caiman/estimates.py` file.

Replace the `~/.local/share/mamba/envs/mescore/lib/python3.10/site-packages/mesmerize_core/algorithms/cnmf.py` file with the `~/caImageAnalysis/caImageAnalysis/replace/mesmerize_core/cnmf.py` file.

Replace the `~/.local/share/mamba/envs/mescore/lib/python3.10/site-packages/mesmerize_core/algorithms/mcorr.py` file with the `~/caImageAnalysis/caImageAnalysis/replace/mesmerize_core/mcorr.py` file.

## How to Use

To run the `.ipynb` Jupyter notebooks:

1. Activate the `mescore` environment:

    ```bash
    micromamba activate mescore
    ```

2. Launch Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

3. Open the `.ipynb` file and run the code.

## Notes

- If you encounter any issues, ensure that all dependencies in the `environment.yml` file were successfully installed.