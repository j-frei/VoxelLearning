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
import tensorflow as tf
from dense_3D_spatial_transformer import Dense3DSpatialTransformer
from losses import cc3D
from volumetools import volumeGradients, tfVectorFieldExp, remap3d, meshgrid, upsampling_resample

def __vnet_level__(in_layer, filters, config,remove_last_conv=False):
    if len(filters) == 1:
        return LeakyReLU()(Conv3D(filters=filters[0],kernel_size=3, padding='same')(in_layer))
    else:
        tlayer = LeakyReLU()(Conv3D(filters=filters[0],kernel_size=3, padding='same')(in_layer))
        tlayer = BatchNormalization()(tlayer) if bool(config.get("batchnorm",False)) else tlayer

        down = LeakyReLU()(Conv3D(filters=filters[0],kernel_size=3, strides=2, padding='same')(tlayer))
        down = BatchNormalization()(tlayer) if bool(config.get("batchnorm",False)) else down

        out_deeper = __vnet_level__(down,filters[1:],config)
        if remove_last_conv:
            return out_deeper
        up = LeakyReLU()(Conv3DTranspose(filters=filters[0],kernel_size=3,strides=2,padding='same')(out_deeper))

        tlayer = Concatenate()([up,tlayer])
        tlayer = BatchNormalization()(tlayer) if bool(config.get("batchnorm",False)) else tlayer

        tlayer = LeakyReLU()(Conv3D(filters=filters[0],kernel_size=3, padding='same')(tlayer))
        tlayer = BatchNormalization()(tlayer) if bool(config.get("batchnorm",False)) else tlayer

        return tlayer

def sampling(args):
    z_mean, z_log_sigma = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1:4]
    #flattened_dim = functools.reduce(lambda x,y:x*y,[*dim,3])
    epsilon = tf.reshape(K.random_normal(shape=(batch, *dim, 3),dtype=tf.float32),(batch,*dim,3))
    xout = z_mean + K.exp(z_log_sigma) * epsilon
    return xout

def samplingGaussian(args,n_gaussians,shape):
    z_mean, z_log_sigma,z_weights = args
    gx,gy,gz = meshgrid(shape[0],shape[1],shape[2])
    grid = tf.stack([gx,gy,gz],-1)


    def sampleGaussian(inputs):
        sampled = tf.zeros(shape=(shape[0],shape[1],shape[2],1))
        mu,sig,w = inputs
        dims = np.asarray(shape[0:3])/2.

        for i in range(n_gaussians):
            dis = MultivariateNormal(loc=(mu[i]/dims)+dims,scale_identity_multiplier=sig[i]*np.max(dims))
            sam = w[i]*K.expand_dims(dis.prob(grid),-1)
            sampled = tf.add(tf.identity(sampled),sam)

        return sampled

    res = tf.map_fn(sampleGaussian,elems=[z_mean,z_log_sigma,z_weights],dtype=tf.float32)
    return res

def toDisplacements(args):
    grads = args
    height = K.shape(grads)[1]
    width = K.shape(grads)[2]
    depth = K.shape(grads)[3]

    _grid = tf.reshape(tf.stack(meshgrid(height,width,depth),-1),(1,height,width,depth,3))
    _stacked = tf.tile(_grid,(tf.shape(grads)[0],1,1,1,1))
    grids = tf.reshape(_stacked,(tf.shape(grads)[0],tf.shape(grads)[1],tf.shape(grads)[2],tf.shape(grads)[3],3))

    return tfVectorFieldExp(grads,grids)

def toUpscaleResampled(args):
    velo = args
    return upsampling_resample(velo)


def transformVolume(args):
    x,disp = args
    moving_vol = tf.reshape(x[:,:,:,:,1],(tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2],tf.shape(x)[3],1))
    #transformed_volumes = Dense3DSpatialTransformer()([moving_vol,disp])
    transformed_volumes = remap3d(moving_vol,disp)
    return transformed_volumes

def empty_loss(true_y,pred_y):
    return tf.constant(0.,dtype=tf.float32)

def smoothness_loss(true_y,pred_y):
    dx = tf.abs(volumeGradients(tf.expand_dims(pred_y[:,:,:,:,0],-1)))
    dy = tf.abs(volumeGradients(tf.expand_dims(pred_y[:,:,:,:,1],-1)))
    dz = tf.abs(volumeGradients(tf.expand_dims(pred_y[:,:,:,:,2],-1)))
    return 1e-5*tf.reduce_sum((dx+dy+dz)/(160*200*160*3), axis=[1, 2, 3, 4])

def sampleLoss(true_y,pred_y):
    z_mean = tf.expand_dims(pred_y[:,:,:,:,0],-1)
    z_log_sigma = tf.expand_dims(pred_y[:,:,:,:,1],-1)
    return - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)


def create_model(config):
    input_shape = (*config['resolution'][0:3],2)

    x = Input(shape=input_shape)
    out = __vnet_level__(x,[8,8],config,config['half_res'])
    # down-conv
    mu = Conv3D(3,kernel_size=3, padding='same')(out)
    log_sigma = Conv3D(3,kernel_size=3, padding='same')(out)
    
    if config['half_res']:
        sampled_velocity_maps = Lambda(sampling)([mu,log_sigma])
        sampled_velocity_maps = Lambda(toUpscaleResampled,name="variationalVelocitySampling")(sampled_velocity_maps)
    else:
        sampled_velocity_maps = Lambda(sampling,name="variationalVelocitySampling")([mu,log_sigma])

    z = Concatenate(name='zVariationalLoss')([mu, log_sigma])
    grads = sampled_velocity_maps

    disp = Lambda(toDisplacements,name="manifold_walk")(grads)

    out = Lambda(transformVolume,name="img_warp")([x,disp])

    loss = [empty_loss,cc3D(),smoothness_loss,sampleLoss]
    lossWeights = [0,1.5,0.1,0.2]
    model = Model(inputs=x,outputs=[disp,out,sampled_velocity_maps,z])
    model.compile(optimizer=Adam(lr=1e-4),loss=loss,loss_weights=lossWeights,metrics=['accuracy'])
    return model
