# Changelog

# [0.1.0]

## Added

- Initial reimplementation of the code found in [vpolo](https://github.com/k3yavi/vpolo) used to parse the binary output from Alevin.

## Changed

- Updated code in `read_quants_bin` and `read_fry_bootstraps_bin` for Python >= 3.10
- Replaced non-functional use of `setuptools-rust` with `pyo3/maturin`

## Fixed

- Rust code now works
