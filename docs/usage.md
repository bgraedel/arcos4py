# Usage

To use arcos4py in a project

```
import arcos4py
```

## Basic usage example of main module

To use the main class, generate a new class instance of ARCOS

```
ts = ARCOS(data,["x"], 'time', 'id', 'meas', 'clTrackID')
```
Data has to be a timeseries provided as a pandas DataFrame in the long format. containing at least a measurment column, a frame/index column and an id column.

On this instance different methods can be run to process the input data and to prepare it for detecting and tracking collective events.

### interpolate Measurments
If the measurment column contains missing values, running this method first to interpolate the data is necessary.

```
ts.interpolate_measurements()

```

### Clip measurments to provided quantille range

cliping can be done to remove extreme outliers from the dataset, but it is not necessary

```
ts.clip_meas(clip_low: = 0.001, clip_high=0.999)
```

### Rescale and Binarize the measurment

Rescaling and detrending is optional for the algorithm to work but is recommended. There are thre options available: ['none', 'lm', 'runmed']. rumned is the default.

Binarization is however required by the ARCOS algorithm to detect and track collective event clusters. This is done by setting a threshold (binThr) and defining measurments below this threshold as 0 and above as 1.

```
ts.bin_measurments(smoothK: int = 3, biasK = 5, peakThr = 1,binThr = 1, polyDeg = 1, biasMet = "runmed",)

```

### Detect collective events

```
events_df = ts.trackCollev(eps = 1, minClsz = 1, nPrev = 1)

```

## Perform calculations without main class

All functions from the ARCOS class are also acessible individuall through the tools module, such as:

```
from arcos4py.tools import trackCollev

```

## Additional modules

In addition to the ARCOS algorithm and its helper classes, plots can be generated with the plotting module and basic collective event statistics can be generated using the stats module
