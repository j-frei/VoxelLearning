# py imports
import os
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn
import SimpleITK as sitk
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import vtk
from vtk.util import numpy_support

def normalize(npv):
    mi = npv.min()
    ma = npv.max()
    return (npv - mi) / (ma - mi)


def centerVolume(x_vol):
    '''
    center the non-zero content inside the x_vol volume
    :param x_vol: 3d volume
    :return: centered 3d volume
    '''
    n_axes = 3
    bounds = []

    def checkZeroByAxis(axis=0, row=0):
        idx = tuple([slice(0, -1) if i != axis else row for i in range(len(x_vol.shape))])
        return (x_vol[idx] == 0.).all()

    for axis in range(n_axes):
        b_low, b_high = 0, -1
        while (checkZeroByAxis(axis, b_low)):
            b_low += 1

        while (checkZeroByAxis(axis, b_high)):
            b_high -= 1

        translation = b_low + b_high
        bounds.append(-int(translation / 2))

    return np.roll(x_vol, bounds, tuple(range(len(bounds))))


def readNormalizedVolumeByPath(vol_path):
    vol = sitk.ReadImage(vol_path)
    # vol.GetSize() yields: (176, 208, 176)
    # axes seems to be: vol[x-axis,z-axis,y-axis] OR vol[y-axis,z-axis,x-axis]

    # make isotrophic
    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetReferenceImage(vol)
    resampleFilter.SetInterpolator(sitk.sitkLinear)
    resampleFilter.SetTransform(sitk.Transform())

    # isotrophic spacing 1mm (according to paper)
    resampleFilter.SetOutputSpacing((1, 1, 1))
    # (160,224,192) seems to be the correct sample size
    # -> OASIS1 has axes: vol[x-axis,z-axis,y-axis]
    resampleFilter.SetSize((160, 224, 192))
    # resampleFilter.SetSize((192,224,160))
    isovol = resampleFilter.Execute(vol)

    # convert to numpy array to
    # remove special itk properties (directions, spacing, ...)
    vol_np = sitk.GetArrayFromImage(isovol)

    # reorder axes
    vol_np = np.moveaxis(vol_np, [0, 1, 2], [1, 2, 0])
    vol_np = np.flip(vol_np, 1)
    vol_np = np.flip(vol_np, 0)
    out = centerVolume(normalize(vol_np))
    return out[::4,::4,::4]

def fixAxesITK(np_vol):
    return np.stack((np_vol[:, :, :, 1], np_vol[:, :, :, 0], np_vol[:, :, :, 2]), 3)

def getGrid():
    vol_size=(160,192,224)
    xx = np.arange(vol_size[1])
    yy = np.arange(vol_size[0])
    zz = np.arange(vol_size[2])
    grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)
    return grid