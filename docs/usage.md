# Usage

To use arcos4py in a project

```
import arcos4py
```

## Basic usage example of the main module

To use the main class, generate a new class instance of ARCOS

```
from arcos4py import ARCOS
ts = ARCOS(data,["x"], 'time', 'id', 'meas', 'clTrackID')
```
Data has to be a time-series provided as a pandas DataFrame in the long format, containing at least a measurement column, a frame/index column, and an id column.

### Prepare the input data.

#### interpolate Measurments
If the measurement column contains missing values, running this method first to interpolate the data is necessary.

```
ts.interpolate_measurements()

```

#### Clip measurement to provided quantile range

Clipping can be performed to remove extreme outliers from the dataset, but it is not necessary.

```
ts.clip_meas(clip_low: = 0.001, clip_high=0.999)
```

#### Rescale and Binarize the measurement

Rescaling and detrending are optional for the algorithm to work but recommended. There are three options available: ['none', 'lm', 'runmed']. Rumned is the default.

However, ARCOS requires binarized data to detect and track collective event clusters. Binarization is done by setting a threshold (binThr) and defining measurement below this threshold as 0 and above as 1.

```
ts.bin_measurements(smoothK: int = 3, biasK = 5, peakThr = 1,binThr = 1, polyDeg = 1, biasMet = "runmed",)

```

### Detect collective events

```
events_df = ts.trackCollev(eps = 1, minClsz = 1, nPrev = 1)

```

## Perform calculations without main class

All functions from the ARCOS class are also accessible individually through the tools module, such as:

```
from arcos4py.tools import trackCollev

```

## Additional modules

In addition to the ARCOS algorithm and its helper classes, plots are generated with the plotting module, collective event statistics using the stats module.
Please see the [Modules Page](api.md) for further details.
