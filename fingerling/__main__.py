from struct import Struct
from pathlib import Path
import numpy as np
import pandas as pd
import gzip
import sys
from scipy.sparse import csr_matrix
from fingerling.fingerling import read_eds


def read_quants_bin(
    base_location: Path | str, clipped: bool = False, mtype="data", rmode="rust"
):
    """
    Read the quants Sparse Binary output of Alevin and generates a dataframe
    Parameters
    ----------
    base_location: string
        Path to the folder containing the output of the alevin run
    clipped: bool (default False)
        Clip off all zero rows and columns
    mtype: "[data(default), tier, var, mean]"
        Alevin's matrix type to load into memory
    """
    base_location = (
        base_location if isinstance(base_location, Path) else Path(base_location)
    )
    if not base_location.is_dir():
        msg = f"{base_location} is not a directory"
        raise ValueError(msg)

    if not base_location.exists():
        msg = f"{base_location} directory doesn't exist"
        raise FileNotFoundError(msg)

    data_type = "f"
    match mtype:
        case "data":
            quant_file = base_location.joinpath("quants_mat.gz")
        case "tier":
            data_type = "B"
            quant_file = base_location.joinpath("quants_tier_mat.gz")
        case "mean":
            quant_file = base_location.joinpath("quants_mean_mat.gz")
        case "var":
            quant_file = base_location.joinpath("quants_var_mat.gz")
        case _:
            msg = f"wrong mtype: {mtype}"
            raise ValueError(msg)

    if not quant_file.exists():
        msg = f"quant file {quant_file} doesn't exist"
        raise FileNotFoundError(msg)

    if mtype in ["mean", "var"]:
        cb_file = base_location.joinpath("quants_boot_rows.txt")
    else:
        cb_file = base_location.joinpath("quants_mat_rows.txt")

    if not cb_file.exists():
        msg = f"The cell barcode file expected at {cb_file} doesn't exist"
        raise FileNotFoundError(msg)

    feature_file = base_location.joinpath("quants_mat_cols.txt")

    if not feature_file.exists():
        msg = f"The feature header expected at {feature_file} doesn't exist"
        raise FileNotFoundError(msg)

    cb_names = pd.read_csv(cb_file, header=None)[0].to_numpy()
    feature_names = pd.read_csv(feature_file, header=None)[0].to_numpy()
    num_features = len(feature_names)
    num_cbs = len(cb_names)
    num_entries = int(np.ceil(num_features / 8))

    if rmode == "rust":
        print(f"Using rust mode with {num_cbs} rows and {num_features} columns")
        mat = read_eds(str(quant_file), num_cbs, num_features)
        umi_matrix = csr_matrix(
            (mat[2], mat[1], mat[0]), shape=(num_cbs, num_features)
        ).todense()
    else:
        with gzip.open(quant_file) as f:
            line_count = 0
            tot_umi_count = 0
            umi_matrix = []

            header_struct = Struct("B" * num_entries)
            while True:
                line_count += 1
                # TODO: this could (should?) be done with tqdm
                if line_count % 100 == 0:
                    print(f"\r Done reading {line_count} cells")
                    sys.stdout.flush()
                try:
                    exp_counts = header_struct.unpack_from(f.read(header_struct.size))
                    num_exp_genes = sum(
                        bin(exp_count).count("1") for exp_count in exp_counts
                    )
                    data_struct = Struct(data_type * num_exp_genes)
                    sparse_cell_counts_vec = list(
                        data_struct.unpack_from(f.read(data_struct.size))
                    )[::-1]
                    cell_umi_counts = sum(sparse_cell_counts_vec)

                except Exception:
                    print(f"\nRead total {str(line_count - 1)} cells")
                    print(f"Found total {str(tot_umi_count)} reads")
                    break

                if cell_umi_counts > 0.0:
                    tot_umi_count += cell_umi_counts

                    cell_counts_vec = []
                    for exp_count in exp_counts:
                        for bit in format(exp_count, "08b"):
                            if len(cell_counts_vec) >= num_features:
                                break

                            if bit == "0":
                                cell_counts_vec.append(0.0)
                            else:
                                abund = sparse_cell_counts_vec.pop()
                                cell_counts_vec.append(abund)

                    if len(sparse_cell_counts_vec) > 0:
                        print("Failure in consumption of data")
                        print(f"left with {len(sparse_cell_counts_vec)} entry(ies)")
                    umi_matrix.append(cell_counts_vec)
                else:
                    msg = "Found a CB with no read count, something is wrong"
                    raise ValueError(msg)

    alv = pd.DataFrame(umi_matrix)
    alv.columns = feature_names
    alv.index = cb_names
    if clipped:
        alv = alv.loc[:, (alv != 0).any(axis=0)]

    return alv
