# Pycewise

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4581363.svg)](https://doi.org/10.5281/zenodo.4581363)

## Installation

```bash
pip install pycewise
```

### Optional installation requirements

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

df = pandas.read_csv('test_data/memcpy_small.csv')
reg = compute_regression(df['size'], df['duration'], mode='log')
print(reg)
```

For more advanced usage, see the [notebooks](https://github.com/Ezibenroc/pycewise/tree/master/notebooks).
