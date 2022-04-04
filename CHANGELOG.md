# Changelog

## [0.1.1] - 2022-04-04
### Added
- More plotting functionallity to the plotting module.
    - Measurment density plot
    - Tracklength histogram
    - Position/T plot
    - Collective event statistcs plot

### Changed
- Interpolation class in tools now uses pandas.interpolate to interpolate missing values.
- Interpolation now interpolates all values in all columns of the dataframe.
- Improved usage section in the documentation.

### Fixed 
- Bug in trackCollev class that would lead to an error message in some cases.
- Spelling in docstrings.

## [0.1.0] - 2022-03-26
### Added
- First release on PyPI.
