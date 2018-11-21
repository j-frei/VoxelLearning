import SimpleITK as sitk
import functools
from functools import partial
import tensorflow as tf
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from tensorflow import keras
import numpy as np
from keras.models import Model
from keras.layers import Conv3D, Conv3DTranspose, Dense, BatchNormalization, Input, Concatenate, UpSampling3D, \
    MaxPool3D, K, Flatten, Reshape, Lambda, LeakyReLU
from tensorflow.contrib.distributions import MultivariateNormalDiag as MultivariateNormal
from tensorflow.python.ops.losses.util import add_loss
#from GroupNorm import GroupNormalization
from dense_3D_spatial_transformer import Dense3DSpatialTransformer
from losses import cc3D
from volumetools import volumeGradients, tfVectorFieldExp, upsample, invertDisplacements, remap3d
import DiffeomorphicRegistrationNet
from Preprocessing import readNormalizedVolumeByPath
import DataGenerator

def generateAvgFromVolumes(vol_center,volumes,model):
    session = tf.Session()

    model_config = {
        'batchsize':1,
        'split':0.9,
        'validation':0.1,
        'half_res':True,
        'epochs': 200,
        'groupnorm':True,
        'GN_groups':32,
        'atlas': 'atlas.nii.gz',
        'model_output': 'model.pkl',
        'exponentialSteps': 7,
    }

    atlas,itk_atlas = DataGenerator.loadAtlas(model_config)

    m = DiffeomorphicRegistrationNet.create_model(model_config)
    m.load_weights(model)
    shapes = atlas.squeeze().shape

    print("First is : {}".format(vol_center))
    vol_first = vol_center
    np_vol_center = readNormalizedVolumeByPath(vol_first,itk_atlas).reshape(1,*shapes).astype(np.float32)

    velocities = []
    for vol in volumes:
        #np_atlas = atlas.reshape(1,*shapes).astype(np.float32)
        np_vol = readNormalizedVolumeByPath(vol,itk_atlas).reshape(1,*shapes).astype(np.float32)

        np_stack = np.empty(1*shapes[0]*shapes[1]*shapes[2]*2,dtype=np.float32).reshape(1,*shapes,2)
        np_stack[:,:,:,:,0] = np_vol
        np_stack[:,:,:,:,1] = np_vol_center

        #tf_stack = tf.convert_to_tensor(np_stack)
        predictions = m.predict(np_stack)
        velocity = predictions[2][0,:,:,:,:]
        velocities.append(velocity)

    # compute avg velocities
    avg_velocity = np.zeros(int(1*shapes[0]/2*shapes[1]/2*shapes[2]/2*3),dtype=np.float32).reshape(1,*[ int(s/2) for s in shapes ],3)
    for v in velocities:
        avg_velocity += v
    avg_velocity /= float(len(velocities))

    # apply squaring&scaling
    steps = model_config['exponentialSteps']
    tf_velo = tf.convert_to_tensor(avg_velocity.reshape(1,*[ int(s/2) for s in shapes ],3))
    tf_vol_center = tf.convert_to_tensor(np_vol_center.reshape(1,*shapes,1))

    x, y, z = K.int_shape(tf_velo)[1:4]

    # clip too large values:
    v_max = 0.5 * (2 ** steps)
    v_min = -v_max
    velo = tf.clip_by_value(tf_velo, v_min, v_max)

    # ij indexing doesn't change (x,y,z) to (y,x,z)
    grid = tf.expand_dims(tf.stack(tf.meshgrid(
        tf.linspace(0., x - 1., x),
        tf.linspace(0., y - 1., y),
        tf.linspace(0., z - 1., z)
        , indexing='ij'), -1),
        0)

    # replicate along batch size
    stacked_grids = tf.tile(grid, (tf.shape(velo)[0], 1, 1, 1, 1))

    displacement = tfVectorFieldExpHalf(velo,stacked_grids,n_steps=steps)
    displacement_highres = toUpscaleResampled(displacement)
    # warp center volume
    new_warped = remap3d(tf_vol_center,displacement_highres)
    with session.as_default():
        new_volume = new_warped.eval(session=session).reshape(*shapes)

    vol_dirs = np.array(itk_atlas.GetDirection()).reshape(3, 3)
    # reapply directions
    warp_np = np.flip(new_volume,[ a for a in range(3) if vol_dirs[a,a] == -1.])
    # prepare axes swap from xyz to zyx
    warp_np = np.transpose(warp_np,(2,1,0))
    # write image
    warp_img = sitk.GetImageFromArray(warp_np)
    warp_img.SetOrigin(itk_atlas.GetOrigin())
    warp_img.SetDirection(itk_atlas.GetDirection())
    sitk.WriteImage(warp_img,"new_volume.nii.gz")

def toUpscaleResampled(args):
    channel_x = args[:,:,:,:,0]
    channel_y = args[:,:,:,:,1]
    channel_z = args[:,:,:,:,2]
    upsampled_x = upsample(tf.expand_dims(channel_x,-1))
    upsampled_y = upsample(tf.expand_dims(channel_y,-1))
    upsampled_z = upsample(tf.expand_dims(channel_z,-1))
    result = tf.squeeze(tf.stack([upsampled_x,upsampled_y,upsampled_z],4),5)
    return result

def tfVectorFieldExpHalf(grad, grid,n_steps):
    N = n_steps

    shapes = tf.shape(grad)
    batch_size, size_x, size_y, size_z, channels = shapes[0], *K.int_shape(grad)[1:5]

    id_x = tf.reshape(grid[:, :, :, :, 0], [batch_size, size_x, size_y, size_z, 1])
    id_y = tf.reshape(grid[:, :, :, :, 1], [batch_size, size_x, size_y, size_z, 1])
    id_z = tf.reshape(grid[:, :, :, :, 2], [batch_size, size_x, size_y, size_z, 1])

    ux = grad[:, :, :, :, 0]
    uy = grad[:, :, :, :, 1]
    uz = grad[:, :, :, :, 2]

    dvx = ux / (2.0 ** N)
    dvy = uy / (2.0 ** N)
    dvz = uz / (2.0 ** N)
    dvx = id_x + tf.reshape(dvx, [batch_size, size_x, size_y, size_z, 1])
    dvy = id_y + tf.reshape(dvy, [batch_size, size_x, size_y, size_z, 1])
    dvz = id_z + tf.reshape(dvz, [batch_size, size_x, size_y, size_z, 1])

    for n in range(0, (N - 1) - 1):
        cache_tf = tf.stack([dvx - id_x, dvy - id_y, dvz - id_z], 4)
        cache_tf = tf.reshape(cache_tf, [batch_size, size_x, size_y, size_z, 3])

        dvx = remap3d(dvx, cache_tf) + tf.expand_dims(cache_tf[:,:,:,:,0],-1)
        dvy = remap3d(dvy, cache_tf) + tf.expand_dims(cache_tf[:,:,:,:,1],-1)
        dvz = remap3d(dvz, cache_tf) + tf.expand_dims(cache_tf[:,:,:,:,2],-1)

    ox = dvx - id_x
    oy = dvy - id_y
    oz = dvz - id_z
    out = tf.reshape(tf.stack([ox, oy, oz], 4), [batch_size, size_x, size_y, size_z, 3])
    return out

if __name__ == '__main__':
    vols_path = "/data/johann/datasets_prepared/OASIS1_*.nii.gz"
    from glob import glob
    all_vols = list(glob(vols_path))
    vol_centered = all_vols[0]
    vols = all_vols[1:][:20]

    generateAvgFromVolumes(vol_centered,vols,"templatemodel.hdf5")
