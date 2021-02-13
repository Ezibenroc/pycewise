# Pycewise

[![Coverage Status](https://coveralls.io/repos/github/Ezibenroc/pycewise/badge.svg?branch=master)](https://coveralls.io/github/Ezibenroc/pycewise?branch=master)

## Installation

### From a wheel (recommended)

```bash
pip install pycewise
```

### Optional requirements

The main functionnality of this package (computing a segmented linear regression) can be used without any third-party code.

For additional features, the following packages should be installed (`pip install <package_name>`):

- numpy
- statsmodels
- jupyter
- matplotlib
- graphviz
- coverage
- mock
- palettable


## Usage

Basic example:

```python
from pycewise import *
import pandas

df = pandas.read_csv('test_data/ringrong_loopback.csv').groupby('size').mean().reset_index()
reg = compute_regression(df['size'], df['duration'], mode='log')
print(reg)
```

For more advanced usage, see the [notebooks](https://github.com/Ezibenroc/pycewise/tree/master/notebooks).
