"""
Reflectance Reconstruction from Tristimulus Values (Prep Fcn)
=============================================================
Computes two arrays needed for method_2.py.

Version 2020_01_24 Scott Allen Burns
"""

import numpy as np

def method_2_prep(cmfs, w):
    """
    Computes two arrays in preparation for the function
    method_2.py. For a given illuminant, method_2_prep
    needs to be called only once. Multiple calls to
    method_2 may then follow for various tristimulus values.

    Parameters
    ----------
    cmfs: nx3 array of color matching functions, where n
          is the number of wavelength bins (rows) of cmfs.
    w: an n-element vector of relative illuminant magnitudes,
          scaled arbitrarily.

    Returns
    -------
    d: nxn array of finite differencing constants.
    cmfs_w: nx3 array of illuminant-w-referenced CMFs.

    Notes
    -----
    This function is identical to method_3_prep.

    Acknowledgements
    ----------------
    Thanks to Adam J. Burns and Mark (kram1032) for assistance
    with translating from MATLAB to Python.

    """

    # determine number of wavelength bins
    n = cmfs.shape[0]

    # build tri-diagonal array of finite differencing constants
    d = np.eye(n, n, k=-1)*-2 + np.eye(n, n)*4 + np.eye(n, n, k=1)*-2
    d[0, 0] = 2
    d[n-1, n-1] = 2

    # build illuminant-w-referenced CMFs
    w1 = np.squeeze(w)
    w_norm = w1/(w1.dot(cmfs[:, 1]))
    cmfs_w = np.diag(w_norm).dot(cmfs)

    return d, cmfs_w