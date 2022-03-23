# arcos4py


[![pypi](https://img.shields.io/pypi/v/arcos4py.svg)](https://pypi.org/project/arcos4py/)
[![python](https://img.shields.io/pypi/pyversions/arcos4py.svg)](https://pypi.org/project/arcos4py/)
[![Build Status](https://github.com/bgraedel/arcos4py/actions/workflows/dev.yml/badge.svg)](https://github.com/bgraedel/arcos4py/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/bgraedel/arcos4py/branch/main/graphs/badge.svg)](https://codecov.io/github/bgraedel/arcos4py)



A python package to detect collective spatio-temporal phenomena


* Documentation: <https://bgraedel.github.io/arcos4py>
* GitHub: <https://github.com/bgraedel/arcos4py>
* PyPI: <https://pypi.org/project/arcos4py/>
* Free software: MIT


## Features

Automated Recognition of Collective Signalling (arcos_py) is a python port of a R package (https://github.com/dmattek/ARCOS
) to identify collective spatial events in time series data.
The software identifies collective protein activation in 2- and 3D cell cultures over time. Such collective waves have been recently identified in various biological systems.
They have been demonstrated to play an important role in the maintenance of epithelial homeostasis (Gagliardi et al., 2020, Takeuchi et al., 2020, Aikin et al., 2020),
in the acinar morphogenesis (Ender et al., 2020), osteoblast regeneration (De Simone et al., 2021), and in the coordination of collective cell migration (Aoki et al., 2017, Hino et al., 2020).

Despite its focus on cell signalling, the framework can be also applied to other spatially correlated phenomena that occur over time.

Data Format
-----------
Time series should be arranged in long format, where each row defines object's location, time, and optionally the measurement value.

ARCOS defines an ARCOS object on which several class methods can be used to prepare the data and calculate collective events.
Optionally the objects used in the ARCOS class can be used individually by importing them from arcos.tools

Installation
------------
The arcos python package can be installed with:

        pip install arcos

## Credits

The original ARCOS algorithm was created by Maciej Dobrzynski (https://github.com/dmattek).
This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.


This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
