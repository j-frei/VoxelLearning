from functools import partial
import tensorflow as tf
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from tensorflow import keras
import numpy as np
from keras import Model
from keras.layers import Conv3D, Conv3DTranspose, Dense, BatchNormalization, Input, Concatenate, UpSampling3D, \
    MaxPool3D, K, Flatten, Reshape, Lambda
from tensorflow.contrib.distributions import MultivariateNormalDiag as MultivariateNormal
from tensorflow.python.ops.losses.util import add_loss

from dense_3D_spatial_transformer import Dense3DSpatialTransformer
from losses import cc3D
from volumetools import volumeGradients, tfVectorFieldExp

def __vnet_level__(in_layer, filters, config):
    if len(filters) == 1:
        return Conv3D(filters=filters[0],kernel_size=3,activation='relu',padding='same')(in_layer)
    else:
        tlayer = Conv3D(filters=filters[0],kernel_size=3, activation='relu', padding='same')(in_layer)
        tlayer = BatchNormalization()(tlayer) if bool(config.get("batchnorm",False)) else tlayer

        tlayer = Conv3D(filters=filters[0],kernel_size=3, activation='relu', padding='same')(tlayer)
        tlayer = BatchNormalization()(tlayer) if bool(config.get("batchnorm",False)) else tlayer

        down = MaxPool3D(pool_size=2)(tlayer)

        out_deeper = __vnet_level__(down,filters[1:],config)
        up = Conv3D(filters[0], 3, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(out_deeper))

        tlayer = Concatenate()([up,tlayer])

        tlayer = Conv3D(filters=filters[0],kernel_size=3, activation='relu', padding='same')(tlayer)
        tlayer = BatchNormalization()(tlayer) if bool(config.get("batchnorm",False)) else tlayer

        tlayer = Conv3D(filters=filters[0],kernel_size=3, activation='relu', padding='same')(tlayer)
        tlayer = BatchNormalization()(tlayer) if bool(config.get("batchnorm",False)) else tlayer

        return tlayer

def sampling(args):
    z_mean, z_log_sigma = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1:4]
    epsilon = K.random_normal(shape=(batch, *dim, 1),dtype=tf.float32)
    xout = z_mean + K.exp(z_log_sigma) * epsilon
    return xout

def _meshgrid(height, width, depth):
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                    tf.transpose(tf.expand_dims(tf.linspace(0.0,
                                                            tf.cast(width, tf.float32)-1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0,
                                               tf.cast(height, tf.float32)-1.0, height), 1),
                    tf.ones(shape=tf.stack([1, width])))

    x_t = tf.tile(tf.expand_dims(x_t, 2), [1, 1, depth])
    y_t = tf.tile(tf.expand_dims(y_t, 2), [1, 1, depth])

    z_t = tf.linspace(0.0, tf.cast(depth, tf.float32)-1.0, depth)
    z_t = tf.expand_dims(tf.expand_dims(z_t, 0), 0)
    z_t = tf.tile(z_t, [height, width, 1])

    return x_t, y_t, z_t


def samplingGaussian(args,n_gaussians,shape):
    z_mean, z_log_sigma,z_weights = args
    gx,gy,gz = _meshgrid(shape[0],shape[1],shape[2])
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

    _grid = tf.reshape(tf.stack(_meshgrid(height,width,depth),-1),(1,height,width,depth,3))
    _stacked = tf.tile(_grid,(tf.shape(grads)[0],1,1,1,1))
    grids = tf.reshape(_stacked,(tf.shape(grads)[0],tf.shape(grads)[1],tf.shape(grads)[2],tf.shape(grads)[3],3))

    return tfVectorFieldExp(tf.multiply(grads,-1.),grids)


def transformVolume(args):
    x,disp = args
    moving_vol = tf.reshape(x[:,:,:,:,1],(tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2],tf.shape(x)[3],1))
    transformed_volumes = Dense3DSpatialTransformer()([moving_vol,disp])
    return transformed_volumes

def empty_loss(true_y,pred_y):
    return tf.constant(0.,dtype=tf.float32)

def sampleLoss(true_y,pred_y):
    z_mean = tf.expand_dims(pred_y[:,:,:,:,0],-1)
    z_log_sigma = tf.expand_dims(pred_y[:,:,:,:,1],-1)
    return - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)


def create_model(input_shape):
    config = {'batchnorm':False}
    n_gaussians = 20
    x = Input(shape=input_shape)
    out = __vnet_level__(x,[32,32],config)
    # down-conv
    mu = Conv3D(1,kernel_size=3, padding='same')(out)
    log_sigma = Conv3D(1,kernel_size=3, padding='same')(out)

    sampled_velocity_maps = Lambda(sampling,name="variationalVelocitySampling")([mu,log_sigma])

    #z = Lambda(lambda args: tf.stack([args[0],args[1]],axis=4), name='zVariationalLoss')([mu, log_sigma])
    z = Concatenate(name='zVariationalLoss')([mu, log_sigma])

    grads = Lambda(volumeGradients,name="gradients")(sampled_velocity_maps)

    disp = Lambda(toDisplacements,name="manifold_walk1")(grads)
    disp = Lambda(toDisplacements,name="manifold_walk2")(disp)
    disp = Lambda(toDisplacements,name="manifold_walk3")(disp)
    disp = Lambda(toDisplacements,name="manifold_walk4")(disp)
    disp = Lambda(toDisplacements,name="manifold_walk5")(disp)
    disp = Lambda(toDisplacements,name="manifold_walk6")(disp)
    disp = Lambda(toDisplacements,name="manifold_walk7")(disp)

    out = Lambda(transformVolume,name="img_warp")([x,disp])

    #loss = [empty_loss,cc3D(),empty_loss,empty_loss,empty_loss]
    loss = [empty_loss,cc3D(),sampleLoss]
    #model = Model(inputs=x,outputs=[disp,out,mu,log_sigma,gaussian_scale])
    model = Model(inputs=x,outputs=[disp,out,z])
    model.compile(optimizer=Adam(lr=1e-4),loss=loss,metrics=['accuracy'])
    return model