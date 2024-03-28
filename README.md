# fingerling

The output of [`salmon alevin`](https://github.com/COMBINE-lab/salmon) is a binary sparse matrix.  [`fingerling`](https://en.wikipedia.org/wiki/Juvenile_fish) is a python module for reading that output into a `pandas` dataframe suitable for downstream analysis.

Code is derived from [`k3yavi/vpolo`](https://github.com/k3yavi/vpolo), updated to use maturin for Python/Rust integration.

## Installation

### Requirements:

`fingerling` requires Python 3.10 (or greater) as well as:
```
pandas
numpy
scipy
```

Additionally, since fingerling contains a submodule written in Rust, a working installation of Rust is required. See instructions for installing Rust [here](https://www.rust-lang.org/tools/install).

Install fingerling from this repository using:
```
pip install git+https://github.com/milescsmith/fingerling
```

If you wish to download the source and build it locally, you will also need [`maturin`](https://www.maturin.rs/). Prior to installing the local files, you will need to run `maturin build` followed by `pip install .target/wheels/fingerling-X.X.X-cp3XX-cp3XX-OS_VERSION_PLATFORM.whl`, where `X.X.X` corresponds to the package version, `cp3XX` the python version, and `OS_VERSION_PLATFORM` your operating system and version.  The actual name and location for the wheel should be given by `maturin` at the end of the build.

## Basic Usage

Currently, `fingerling` is not setup to be used as a standalone executable, but instead used as part of an analysis pipeline in something like a Jupyter notebook.  The `read_quants_bin` function returns a sample-by-feature dataframe with an index corresponding to the cell barcodes and column names corresponding to the feature names and is suitable for use in creating an [`AnnData`](https://anndata.readthedocs.io/en/latest/) object:

```
import scanpy as sc
from fingerling import read_quants_bin

cite_df = read_quants_bin("/PATH/TO/ALEVIN_OUTPUT")

adata = sc.Anndata(X=cite_df)
```