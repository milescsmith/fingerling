import importlib.resources as ir
import pytest
from pathlib import Path
from fingerling import read_quants_bin
import pandas as pd
import pandas.testing as pdt


@pytest.fixture
def alevin_dir() -> Path:
    return ir.files("tests").joinpath("data", "alevin")


@pytest.fixture
def expected_df() -> pd.DataFrame:
    return pd.read_csv(
        ir.files("tests").joinpath("data", "expected_read_quants_bin_df.csv"),
        index_col=0,
    )


def test_read_quants_bin(expected_df, alevin_dir):
    alevin_res = read_quants_bin(alevin_dir)
    pdt.assert_frame_equal(expected_df, alevin_res)
