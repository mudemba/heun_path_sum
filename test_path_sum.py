import numpy as np
from heun_path_sum import subdivide_domain


def test_subdivide_domain():
    domain = np.linspace(1, 100, 100)
    target = np.split(domain, 5)
    subdivided_domain = subdivide_domain(
        domain, max_sub_points=20, max_sub_width=19)

    assert subdivided_domain is target
