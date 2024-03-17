# Documentation

## Installation

Copy [`quant.py`](./code/quant.py) into the same directory as your code.

## Basic Usage

To use QUANT:
* import QUANT and the ExtraTreesClassifier (from scikit-learn);
* load the data;
* transform the data using QUANT;
* train the classifier; and
* predict the class(es) for the test data.

**Import QUANT and ExtraTreesClassifier**

```python
from quant import Quant
from sklearn.ensemble import ExtraTreesClassifier
```

**Load Data**

*Note: QUANT uses PyTorch.  The input time series must be in PyTorch tensor format.*

```python
X_tr, Y_tr = torch.tensor(np.load("./X_tr.npy")).float(), np.load("./Y_tr.npy")
X_te, Y_te = torch.tensor(np.load("./X_te.npy")).float(), np.load("./Y_te.npy")
```

**Transform Training Set**

```python
quant = Quant()

X_tr_transform = quant.fit_transform(X_tr, Y_tr)
```

**Fit Classifier**

*Note: QUANT is intended to be used with an extremely randomised trees classifier, with 200 trees, and considering 10% of the total number of features at each split (and using 'entropy' as the splitting criterion).*

```python
classifier = \
ExtraTreesClassifier(
    n_estimators = 200,
    max_features = 0.1,
    criterion = "entropy",
    n_jobs = -1,
)
classifier.fit(X_tr_transform, Y_tr)
```

**Transform Test Set**

```python
X_te_transform = quant.transform(X_te)
```

**Predict**

```python
predictions = classifier.predict(X_te_transform)
```

## Configuring QUANT

QUANT has two main parameters: (a) **depth**; and (b) **the number of quantiles per interval**.  These parameters are are intended to be kept at their default values, but can be set manually if desired.

**Depth**

QUANT uses dyadic intervals (i.e., forms intervals by recursively splitting the input time series in half).  The depth parameter controls the number of times this splitting occurs (and, consequently, the total number of intervals).  In particular, for a depth of $d$, input time series are split into $2^{0} + 2^{1} + ... + 2^{d-1}$ intervals.  The default value for the depth parameter is 6.

The depth parameter is controlled by the `depth` argument in the `Quant()` constructor.  E.g., for a depth of $2$ you would specify `Quant(depth = 2)`.

**Number of Quantiles per Interval**

QUANT computes $m / v$ quantiles per interval, where $m$ is interval length.  By default, QUANT computes $m / 4$ (i.e., $v = 4$) quantiles per interval.  E.g., for an interval of length $64$, QUANT computes $64 / 4 = 16$ quantiles.

This parameter is controlled by the `div` argument in the `Quant()` constructor.  E.g., for $m / 8$ quantiles per interval (i.e., $v = 8$) you would specify `Quant(div = 8)`.

## Reproducing the Experiments

### UCR Archive

[TBA]

### Sensitivity Analysis

[TBA]