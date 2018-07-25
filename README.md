# Pytree

[![Build Status](https://travis-ci.org/Ezibenroc/pytree.svg?branch=master)](https://travis-ci.org/Ezibenroc/pytree)
[![Coverage Status](https://coveralls.io/repos/github/Ezibenroc/pytree/badge.svg?branch=master)](https://coveralls.io/github/Ezibenroc/pytree?branch=master)

## Installation

### From a wheel (recommended)

```bash
pip install https://github.com/Ezibenroc/pytree/releases/download/0.0.5/pytree-0.0.5-py3-none-any.whl
```

Replace the two occurences of the version number in the URL by the version you wish to install.

### From source

```bash
git clone https://github.com/Ezibenroc/pytree.git
cd pytree
python3 setup.py install --user
```

### Optional requirements

The main functionnality of this package (computing a segmented linear regression) can be used without any third-party code.

For additional features, the following packages should be installed (`pip install <package_name>`):

- statsmodels
- jupyter
- matplotlib
- graphviz
- coverage
- mock
- palettable


## Usage

See the notebooks [demo.ipynb](demo.ipynb) and [advanced_features.ipynb](advanced_features.ipynb).
