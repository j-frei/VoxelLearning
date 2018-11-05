import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from volumetools import remap3d, upsample, invertDisplacements
session = tf.Session()

#vol = sitk.ReadImage("OASIS1_0001.nii.gz")
vol = sitk.ReadImage("OASIS1_0001.nii.gz")
trf = sitk.ReadImage("OASIS1_0001-output_0Warp.nii.gz")

# sitk has xyz in GetPixel
vol_np = sitk.GetArrayFromImage(vol) #[::2,::2,::2]
# swap axes from default zyx (numpy) to xyz
vol_np = np.transpose(vol_np,(2,1,0))
# apply directions
vol_dirs = np.array(vol.GetDirection()).reshape(3,3)
vol_np = np.flip(vol_np,[ a for a in range(3) if vol_dirs[a,a] == -1.])

# sitk has xyz in GetPixel
trf_np = sitk.GetArrayFromImage(trf) #[::2,::2,::2]
# swap axes from default zyx (numpy) to xyz
trf_np = np.transpose(trf_np,(2,1,0,3))
# apply directions
trf_dirs = np.array(trf.GetDirection()).reshape(3,3)
trf_np = np.flip(trf_np,[ a for a in range(3) if trf_dirs[a,a] == -1.])

# swap axes from xy to ij for spacial transformer: (x,y,z) -> (y,x,z)
#trf_np = np.transpose(trf_np,(1,0,2,3))
#vol_np = np.transpose(vol_np,(1,0,2))

tf_vol = tf.convert_to_tensor(vol_np.reshape(1,*vol_np.shape[0:3],1).astype(np.float32))
tf_trf = tf.convert_to_tensor(trf_np.reshape(1,*trf_np.shape[0:3],3).astype(np.float32))

#warp = upsample(tf_vol)
inv_trf = invertDisplacements(tf_trf)
inv_trf = invertDisplacements(inv_trf)
with session.as_default():
    warp_np = inv_trf.eval(session=session).squeeze() #.reshape(*vol_np.shape,1)

# reapply directions
warp_np = np.flip(warp_np,[ a for a in range(3) if vol_dirs[a,a] == -1.])
# prepare axes swap from xyz to zyx
warp_np = np.transpose(warp_np,(2,1,0,3))
# write image
warp_img = sitk.GetImageFromArray(warp_np)
warp_img.SetOrigin(vol.GetOrigin())
warp_img.SetDirection(vol.GetDirection())
sitk.WriteImage(warp_img,"warp_inv.nii.gz")
