# QUANT

***QUANT: A Minimalist Interval Method for Time Series Classification***

[Data Mining and Knowledge Discovery](https://doi.org/10.1007/s10618-024-01036-9) / [arXiv:2308.00928](https://arxiv.org/abs/2308.00928) (preprint)


> <div align="justify">We show that it is possible to achieve the same accuracy, on average, as the most accurate existing interval methods for time series classification on a standard set of benchmark datasets using a single type of feature (quantiles), fixed intervals, and an 'off the shelf' classifier. This distillation of interval-based approaches represents a fast and accurate method for time series classification, achieving state-of-the-art accuracy on the expanded set of 142 datasets in the UCR archive with a total compute time (training and inference) of less than 15 minutes using a single CPU core.</div>

Please cite as:

```bibtex
@article{dempster_etal_2024,
  author  = {Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
  title   = {{QUANT}: A Minimalist Interval Method for Time Series Classification},
  year    = {2024},
  journal = {Data Mining and Knowledge Discovery},
}
```

## Results

#### UCR Archive (142 Datasets, 30 Resamples)

* [Mean Accuracy + Training/Test Times](./results/accuracy_mean_ucr142.csv)
* [Per Resample](./results/accuracy_resamples_ucr142.csv)

## Requirements

* Python
* PyTorch
* NumPy
* scikit-learn (or similar)

## Code

### [`quant.py`](./code/quant.py)

## Documentation

[Documentation](./doc.md) *[in progress]*

## Examples

```python
from quant import Quant
from sklearn.ensemble import ExtraTreesClassifier

[...] # load data -> torch.float32, [num_examples, 1, length]

transform = Quant()

X_training_transform = transform.fit_transform(X_training, Y_training)
X_test_transform = transform.transform(X_test)

classifier = \
ExtraTreesClassifier(
    n_estimators = 200,
    max_features = 0.1,
    criterion = "entropy",
    n_jobs = -1
)
classifier.fit(X_training_transform, Y_training)

predictions = classifier.predict(X_test_transform)
```

## Acknowledgements

We thank Professor Eamonn Keogh and all the people who have contributed to the UCR time series classification archive.

<div align="center">:zap:</div>