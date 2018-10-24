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
from volumetools import volumeGradients, tfVectorFieldExp, remap3d, upsample

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

def toDisplacements(steps=7):
    def exponentialMap(args):
        grads = args
        x,y,z = K.int_shape(args)[1:4]

        # ij indexing doesn't change (x,y,z) to (y,x,z)
        grid = tf.expand_dims(tf.stack(tf.meshgrid(
            tf.linspace(0.,x-1.,x),
            tf.linspace(0.,y-1.,y),
            tf.linspace(0.,z-1.,z)
            ,indexing='ij'),-1),
        0)

        # replicate along batch size
        stacked_grids = tf.tile(grid,(tf.shape(grads)[0],1,1,1,1))

        res = tfVectorFieldExp(grads,stacked_grids,n_steps=steps)
        return res
    return exponentialMap

def toUpscaleResampled(args):
    channel_x = args[:,:,:,:,0]
    channel_y = args[:,:,:,:,1]
    channel_z = args[:,:,:,:,2]
    upsampled_x = upsample(tf.expand_dims(channel_x,-1))
    upsampled_y = upsample(tf.expand_dims(channel_y,-1))
    upsampled_z = upsample(tf.expand_dims(channel_z,-1))
    result = tf.squeeze(tf.stack([upsampled_x,upsampled_y,upsampled_z],4),5)
    return result

def transformVolume(args):
    x,disp = args
    moving_vol = tf.reshape(x[:,:,:,:,1],(tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2],tf.shape(x)[3],1))
    transformed_volumes = remap3d(moving_vol,disp)
    return transformed_volumes

def empty_loss(true_y,pred_y):
    return tf.constant(0.,dtype=tf.float32)

def smoothness(batch_size):
    def smoothness_loss(true_y,pred_y):
        dx = tf.abs(volumeGradients(tf.expand_dims(pred_y[:,:,:,:,0],-1)))
        dy = tf.abs(volumeGradients(tf.expand_dims(pred_y[:,:,:,:,1],-1)))
        dz = tf.abs(volumeGradients(tf.expand_dims(pred_y[:,:,:,:,2],-1)))

        return tf.reduce_sum((dx+dy+dz)/(functools.reduce(lambda x,y:x*y,K.int_shape(pred_y)[1:5])*batch_size), axis=[1, 2, 3, 4])
    return smoothness_loss

def sampleLoss(true_y,pred_y):
    z_mean = tf.expand_dims(pred_y[:,:,:,:,0],-1)
    z_log_sigma = tf.expand_dims(pred_y[:,:,:,:,1],-1)
    return - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)


def create_model(config):
    input_shape = (*config['resolution'][0:3],2)

    x = Input(shape=input_shape)
    out = __vnet_level__(x,[16,32,32],config,remove_last_conv=config['half_res'])
    # down-conv
    mu = Conv3D(3,kernel_size=3, padding='same')(out)
    log_sigma = Conv3D(3,kernel_size=3, padding='same')(out)    

    sampled_velocity_maps = Lambda(sampling,name="variationalVelocitySampling")([mu,log_sigma])

    z = Concatenate(name='zVariationalLoss')([mu, log_sigma])
    grads = sampled_velocity_maps

    if config['half_res']:
        disp_low = Lambda(toDisplacements(steps=config['exponentialSteps']))(grads)
        # upsample displacement map
        disp_upsampled = Lambda(toUpscaleResampled)(disp_low)
        # we need to fix displacement vectors which are too small after upsampling
        disp = Lambda(lambda dispMap: tf.scalar_mul(2.,dispMap),name="manifold_walk")
    else:
        disp = Lambda(toDisplacements,name="manifold_walk")(grads)

    warped = Lambda(transformVolume,name="img_warp")([x,disp])

    loss = [empty_loss,cc3D(),smoothness(config['batchsize']),sampleLoss]
    lossWeights = [0.,
                   # data term / CC
                   1.0,
                   # smoothness
                   0.002,
                   # loglikelihood
                   0.2
                   ]
    model = Model(inputs=x,outputs=[disp,warped,sampled_velocity_maps,z])
    model.compile(optimizer=Adam(lr=1e-4),loss=loss,loss_weights=lossWeights,metrics=['accuracy'])
    return model
