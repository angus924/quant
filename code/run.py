import argparse
import numpy as np
import pandas as pd
import re
import torch

from quant import Quant

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier

from tqdm import tqdm

# ==============================================================================
# == args ======================================================================
# ==============================================================================

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_path")
parser.add_argument("-o", "--output_path", default = "./")

parser.add_argument("-d", "--datasets")

parser.add_argument("-r", "--num_resamples", type = int, default = 30)
parser.add_argument("-p", "--resample_indices_path")

parser.add_argument("-m", "--depth", type = int, default = 6)
parser.add_argument("-v", "--div", type = int, default = 4)

parser.add_argument("-e", "--num_estimators", type = int, default = 200)
parser.add_argument("-f", "--max_features", default = "0.1")

args = parser.parse_args()

_max_features = float(args.max_features) if args.max_features not in ["sqrt"] else args.max_features

print(f"-- args ".ljust(80, "-"))
[print(f"{k}: {v}") for k, v in vars(args).items()]
print("-" * 80)

# ==============================================================================
# == file handling =============================================================
# ==============================================================================

# from <https://github.com/aeon-toolkit/aeon/blob/main/aeon/datasets/_data_loaders.py>

def _load_header_info(file):
    """Load the meta data from a .ts file and advance file to the data.

    Parameters
    ----------
    file : stream.
        input file to read header from, assumed to be just opened

    Returns
    -------
    meta_data : dict.
        dictionary with the data characteristics stored in the header.
    """
    meta_data = {
        "problemname": "none",
        "timestamps": False,
        "missing": False,
        "univariate": True,
        "equallength": True,
        "classlabel": True,
        "targetlabel": False,
        "class_values": [],
    }
    boolean_keys = ["timestamps", "missing", "univariate", "equallength", "targetlabel"]
    for line in file:
        line = line.strip().lower()
        line = re.sub(r"\s+", " ", line)
        if line and not line.startswith("#"):
            tokens = line.split(" ")
            token_len = len(tokens)
            key = tokens[0][1:]
            if key == "data":
                if line != "@data":
                    raise OSError("data tag should not have an associated value")
                return meta_data
            if key in meta_data.keys():
                if key in boolean_keys:
                    if token_len != 2:
                        raise OSError(f"{tokens[0]} tag requires a boolean value")
                    if tokens[1] == "true":
                        meta_data[key] = True
                    elif tokens[1] == "false":
                        meta_data[key] = False
                elif key == "problemname":
                    meta_data[key] = tokens[1]
                elif key == "classlabel":
                    if tokens[1] == "true":
                        meta_data["classlabel"] = True
                        if token_len == 2:
                            raise OSError(
                                "if the classlabel tag is true then class values "
                                "must be supplied"
                            )
                    elif tokens[1] == "false":
                        meta_data["classlabel"] = False
                    else:
                        raise OSError("invalid class label value")
                    meta_data["class_values"] = [token.strip() for token in tokens[2:]]
        if meta_data["targetlabel"]:
            meta_data["classlabel"] = False
    return meta_data

def _load_data(file, meta_data, replace_missing_vals_with="NaN"):
    """Load data from a file with no header.

    this assumes each time series has the same number of channels, but allows unequal
    length series between cases.

    Parameters
    ----------
    file : stream, input file to read data from, assume no comments or header info
    meta_data : dict.
        with meta data in the file header loaded with _load_header_info

    Returns
    -------
    data: list[np.ndarray].
        list of numpy arrays of floats: the time series
    y_values : np.ndarray.
        numpy array of strings: the class/target variable values
    meta_data :  dict.
        dictionary of characteristics enhanced with number of channels and series length
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and "class_values": [],

    """
    data = []
    n_cases = 0
    n_channels = 0  # Assumed the same for all
    current_channels = 0
    n_timepoints = 0
    y_values = []
    target = False
    if meta_data["classlabel"] or meta_data["targetlabel"]:
        target = True
    for line in file:
        line = line.strip().lower()
        line = line.replace("nan", replace_missing_vals_with)
        line = line.replace("?", replace_missing_vals_with)
        if "timestamps" in meta_data and meta_data["timestamps"]:
            channels = _get_channel_strings(line, target, replace_missing_vals_with)
        else:
            channels = line.split(":")
        n_cases += 1
        current_channels = len(channels)
        if target:
            current_channels -= 1
        if n_cases == 1:  # Find n_channels and length  from first if not unequal
            n_channels = current_channels
            if meta_data["equallength"]:
                n_timepoints = len(channels[0].split(","))
        else:
            if current_channels != n_channels:
                raise OSError(
                    f"Inconsistent number of dimensions in case {n_cases}. "
                    f"Expecting {n_channels} but have read {current_channels}"
                )
            if meta_data["univariate"]:
                if current_channels > 1:
                    raise OSError(
                        f"Seen {current_channels} in case {n_cases}."
                        f"Expecting univariate from meta data"
                    )
        if meta_data["equallength"]:
            current_length = n_timepoints
        else:
            current_length = len(channels[0].split(","))
        np_case = np.zeros(shape=(n_channels, current_length))
        for i in range(0, n_channels):
            single_channel = channels[i].strip()
            data_series = single_channel.split(",")
            data_series = [float(x) for x in data_series]
            if len(data_series) != current_length:
                equal_length = meta_data["equallength"]
                raise OSError(
                    f"channel {i} in case {n_cases} has a different number of "
                    f"observations to the other channels. "
                    f"Saw {current_length} in the first channel but"
                    f" {len(data_series)} in the channel {i}. The meta data "
                    f"specifies equal length == {equal_length}. But even if series "
                    f"length are unequal, all channels for a single case must be the "
                    f"same length"
                )
            np_case[i] = np.array(data_series)
        data.append(np_case)
        if target:
            y_values.append(channels[n_channels])
    if meta_data["equallength"]:
        data = np.array(data)
    return data, np.asarray(y_values), meta_data

def load_from_tsfile(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    return_meta_data=False,
    return_type="auto",
):
    """Load time series .ts file into X and (optionally) y.

    Parameters
    ----------
    full_file_path_and_name : string
        full path of the file to load, .ts extension is assumed.
    replace_missing_vals_with : string, default="NaN"
        issing values in the file are replaces with this value
    return_meta_data : boolean, default=False
        return a dictionary with the meta data loaded from the file
    return_type : string, default = "auto"
        data type to convert to.
        If "auto", returns numpy3D for equal length and list of numpy2D for unequal.
        If "numpy2D", will squash a univariate equal length into a numpy2D (n_cases,
        n_timepoints). Other options are available but not supported medium term.

    Returns
    -------
    data: Union[np.ndarray,list]
        time series data, np.ndarray (n_cases, n_channels, n_timepoints) if equal
        length time series, list of [n_cases] np.ndarray (n_channels, n_timepoints)
        if unequal length series.
    y : target variable, np.ndarray of string or int
    meta_data : dict (optional).
        dictionary of characteristics, with keys
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and "class_values": [],

    Raises
    ------
    IOError if the load fails.
    """
    # Check file ends in .ts, if not, insert
    if not full_file_path_and_name.endswith(".ts"):
        full_file_path_and_name = full_file_path_and_name + ".ts"
    # Open file
    with open(full_file_path_and_name, encoding="utf-8") as file:
        # Read in headers
        meta_data = _load_header_info(file)
        # load into list of numpy
        data, y, meta_data = _load_data(file, meta_data)

    # if equal load to 3D numpy
    if meta_data["equallength"]:
        data = np.array(data)
        if return_type == "numpy2D" and meta_data["univariate"]:
            data = data.squeeze()
    # If regression problem, convert y to float
    if meta_data["targetlabel"]:
        y = y.astype(float)
    if return_meta_data:
        return data, y, meta_data
    return data, y

# ==============================================================================
# == run =======================================================================
# ==============================================================================

datasets = np.loadtxt(f"{args.datasets}", "str")

results = []

for dataset in tqdm(datasets, ncols = 80):

    for i in range(args.num_resamples):

        # ======================================================================

        _X_TR, _Y_TR = load_from_tsfile(f"{args.input_path}/{dataset}/{dataset}_TRAIN.ts")
        _X_TE, _Y_TE = load_from_tsfile(f"{args.input_path}/{dataset}/{dataset}_TEST.ts")

        if i == 0:
            X_TR = _X_TR
            X_TE = _X_TE
            Y_TR = _Y_TR
            Y_TE = _Y_TE
        else:
            _X = np.concatenate((_X_TR, _X_TE))
            _Y = np.concatenate((_Y_TR, _Y_TE))

            indices = np.arange(_X.shape[0])

            indices_TR = np.loadtxt(f"{args.resample_indices_path}/{dataset}/resample{i}Indices_TRAIN.txt", dtype = np.int32)
            indices_TE = np.setdiff1d(indices, indices_TR)

            X_TR = _X[indices_TR]
            X_TE = _X[indices_TE]

            Y_TR = _Y[indices_TR]
            Y_TE = _Y[indices_TE]

        X_TR = torch.tensor(X_TR.squeeze().astype(np.float32)).unsqueeze(1)
        X_TE = torch.tensor(X_TE.squeeze().astype(np.float32)).unsqueeze(1)

        le = LabelEncoder()

        Y_TR = le.fit_transform(Y_TR)
        Y_TE = le.transform(Y_TE)

        # ======================================================================

        quant = Quant(
            depth = args.depth,
            div = args.div,
        )

        Z_TR = quant.fit_transform(X_TR, Y_TR)

        classifier = \
        ExtraTreesClassifier(
            n_estimators = args.num_estimators,
            max_features = _max_features,
            criterion = "entropy",
            n_jobs = -1,
        )

        classifier.fit(Z_TR, Y_TR)

        Z_TE = quant.transform(X_TE)

        accuracy = (classifier.predict(Z_TE) == Y_TE).mean()

        results.append((dataset, i, accuracy))

# == save ======================================================================

results = pd.DataFrame(results)
results.columns = ["dataset", "resample", "accuracy"]
results = results.set_index(["dataset", "resample"])
results.round(6).to_csv(f"{args.output_path}/results.csv")