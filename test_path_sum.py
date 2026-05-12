import numpy as np
from heun_path_sum import subdivide_domain, clip_interval


def test_clip_interval_1():
    """Tests basic functionality of clip_interval"""
    interval = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    target = np.array([1., 2., 3., 4., 5., 6., 7.])

    clipped_interval = clip_interval(interval, 3)

    assert clipped_interval.all() == target.all()


def test_clip_interval_2():
    """Don't clip if the number of points is a multiple of desired number of points"""
    interval = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    clipped_interval = clip_interval(interval, 5)

    assert len(clipped_interval) == 10
