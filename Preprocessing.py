import SimpleITK as sitk
import numpy as np

def resampleImage(img,resolution,spacings):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing(spacings)
    resampler.SetSize(resolution)
    return resampler.Execute(img)

def normalizeImage(img):
    f = sitk.NormalizeImageFilter()
    return f.Execute(img)

def readNormalizedVolumeByPath(vol_path,config):
    in_img = sitk.ReadImage(vol_path)
    resampled = resampleImage(in_img,config.get('resolution',(128,128,128)),config.get('spacings',(1.5,1.5,1.5)))
    normalized = normalizeImage(resampled)
    # fix normalized (-1,1) to (0,1)
    return ((sitk.GetArrayFromImage(normalized) + 1. ) / 2.).astype("float32")
