import numpy as np
import pytest

from arcos4py.tools._cleandata import blockwise_median, remove_image_background


def test_remove_image_background():
    image = np.random.rand(50, 5, 5)

    # Test with valid input
    result = remove_image_background(image, filter_type='gaussian', crop_time_axis=False)
    assert result.shape == (50, 5, 5)

    # Test crop_time_axis
    result_cropped = remove_image_background(image, filter_type='gaussian', crop_time_axis=True)
    assert result_cropped.shape[0] < image.shape[0]

    # Test invalid filter type
    with pytest.raises(ValueError):
        remove_image_background(image, filter_type='invalid_filter')

    # Test invalid dimensions
    with pytest.raises(ValueError):
        remove_image_background(image, dims="TXYZ")

    # Test invalid size
    with pytest.raises(ValueError):
        remove_image_background(image, size=(10, 10))


def test_blockwise_median():
    array = np.random.rand(8, 8)

    # Test valid blockshape
    result = blockwise_median(array, blockshape=(2, 2))
    assert result.shape == (4, 4)

    # Test blockshape with wrong dimensions
    with pytest.raises(AssertionError):
        blockwise_median(array, blockshape=(2, 2, 2))

    # Test blockshape that does not divide cleanly
    with pytest.raises(AssertionError):
        blockwise_median(array, blockshape=(3, 3))
