from mmd_twosample.core import _rbf_dot, _compute_kernel_size, mmdTestBoot, mmdTestGamma
import numpy as np


def test_rbf_dot():
    X1 = np.loadtxt(r"../testdata/X1.txt", skiprows=5)
    Y1 = np.loadtxt(r"../testdata/Y1.txt", skiprows=5)
    Hxy = np.loadtxt(r"../testdata/rbf_X1_Y1.txt", skiprows=5)
    Hxx = np.loadtxt(r"../testdata/rbf_X1_X1.txt", skiprows=5)

    assert len(X1) == 100 and len(Y1) == 100

    assert np.linalg.norm(_rbf_dot(X1, X1, deg=1.5353) - Hxx) < 1e-2
    assert np.linalg.norm(_rbf_dot(X1, Y1, deg=1.5353) - Hxy) < 1e-2


def test_compute_kernel_size():
    X1 = np.loadtxt(r"../testdata/X1.txt", skiprows=5)
    Y1 = np.loadtxt(r"../testdata/Y1.txt", skiprows=5)

    my_sig = _compute_kernel_size(X1, Y1, max_samples=100)
    assert abs(my_sig - 1.5353) < 1e-3


def test_mmdTestBoot():
    X1 = np.loadtxt(r"../testdata/X1.txt", skiprows=5)
    Y1 = np.loadtxt(r"../testdata/Y1.txt", skiprows=5)

    testStat, thresh = mmdTestBoot(X1, Y1, alpha= 0.05, params={'shuff': 1000, 'bootForce': True})

    assert abs(testStat - 45.387) < 1e-3
    assert 2.45 <= thresh <= 2.85
    print(thresh)


def test_mmdTestGamma():
    X1 = np.loadtxt(r"../testdata/X1.txt", skiprows=5)
    Y1 = np.loadtxt(r"../testdata/Y1.txt", skiprows=5)

    testStat, thresh = mmdTestGamma(X1, Y1, alpha= 0.05, params={'shuff': 1000, 'bootForce': True})

    assert abs(testStat - 45.387) < 1e-3
    assert abs(thresh - 3.3464) < 1e-3
    print(thresh)
