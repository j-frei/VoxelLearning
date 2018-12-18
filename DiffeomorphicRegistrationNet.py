import functools
from functools import partial
import tensorflow as tf
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from tensorflow import keras
import numpy as np
from keras.models import Model
from keras.layers import Conv3D, Conv3DTranspose, Dense, BatchNormalization, Input, Concatenate, UpSampling3D, \
    MaxPool3D, K, Flatten, Reshape, Lambda, LeakyReLU, Add, Average
from tensorflow.contrib.distributions import MultivariateNormalDiag as MultivariateNormal
from tensorflow.python.ops.losses.util import add_loss
from GroupNorm import GroupNormalization
from dense_3D_spatial_transformer import Dense3DSpatialTransformer
from losses import cc3D
from volumetools import volumeGradients, tfVectorFieldExp, remap3d, upsample, invertDisplacements, concatenateTransforms

def build_vnet_model(filters,config):
    input_shape = (*config['resolution'][0:3],2)
    x = Input(shape=input_shape)

    def __vnet_level__(in_layer, filters, config,remove_last_conv=False):
        if len(filters) == 1:
            return LeakyReLU()(Conv3D(filters=filters[0],kernel_size=3, padding='same')(in_layer))
        else:
            tlayer = LeakyReLU()(Conv3D(filters=filters[0],kernel_size=3, padding='same')(in_layer))
            tlayer = BatchNormalization()(tlayer) if bool(config.get("batchnorm",False)) else tlayer
            tlayer = GroupNormalization(groups=config.get("GN_groups",0))(tlayer) if bool(config.get("groupnorm",False)) else tlayer

            down = LeakyReLU()(Conv3D(filters=filters[0],kernel_size=3, strides=2, padding='same')(tlayer))
            down = BatchNormalization()(tlayer) if bool(config.get("batchnorm",False)) else down
            tlayer = GroupNormalization(groups=config.get("GN_groups",0))(tlayer) if bool(config.get("groupnorm",False)) else tlayer

            out_deeper = __vnet_level__(down,filters[1:],config)
            if remove_last_conv:
                return out_deeper
            up = LeakyReLU()(Conv3DTranspose(filters=filters[0],kernel_size=3,strides=2,padding='same')(out_deeper))

            tlayer = Concatenate()([up,tlayer])
            tlayer = BatchNormalization()(tlayer) if bool(config.get("batchnorm",False)) else tlayer
            tlayer = GroupNormalization(groups=config.get("GN_groups",0))(tlayer) if bool(config.get("groupnorm",False)) else tlayer

            tlayer = LeakyReLU()(Conv3D(filters=filters[0],kernel_size=3, padding='same')(tlayer))
            tlayer = BatchNormalization()(tlayer) if bool(config.get("batchnorm",False)) else tlayer
            tlayer = GroupNormalization(groups=config.get("GN_groups",0))(tlayer) if bool(config.get("groupnorm",False)) else tlayer

            return tlayer
    y = __vnet_level__(x,filters,config,remove_last_conv=config.get("half_res",False))
    return Model(inputs=x,outputs=y)

def build_sampling_model(config, channels_of_input):
    res = config['resolution'][0:3]
    if config.get('half_res',False):
        res = [ int(axis/2) for axis in res ]

    basic_input_shape = (*res,channels_of_input)
    x = Input(shape=basic_input_shape)

    # down-conv
    mu = Conv3D(3,kernel_size=3, padding='same')(x)
    log_sigma = Conv3D(3,kernel_size=3, padding='same')(x)

    y = [mu,log_sigma]
    return Model(inputs=x,outputs=y)

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
        velo_raw = args
        x,y,z = K.int_shape(args)[1:4]

        # clip too large values:
        v_max = 0.5 * (2**steps)
        v_min = -v_max
        velo = tf.clip_by_value(velo_raw,v_min,v_max)

        # ij indexing doesn't change (x,y,z) to (y,x,z)
        grid = tf.expand_dims(tf.stack(tf.meshgrid(
            tf.linspace(0.,x-1.,x),
            tf.linspace(0.,y-1.,y),
            tf.linspace(0.,z-1.,z)
            ,indexing='ij'),-1),
        0)

        # replicate along batch size
        stacked_grids = tf.tile(grid,(tf.shape(velo)[0],1,1,1,1))

        res = tfVectorFieldExp(velo,stacked_grids,n_steps=steps)
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

def transformVolume1(args):
    x,disp = args
    shape = [tf.shape(x)[0], *K.int_shape(x)[1:4], 1]
    moving_vol = tf.reshape(x[:,:,:,:,1],shape)
    transformed_volumes = remap3d(moving_vol,disp)
    return transformed_volumes

def transformVolume2(args):
    x,disp = args
    shape = [tf.shape(x)[0], *K.int_shape(x)[1:4], 1]
    moving_vol = tf.reshape(x[:,:,:,:,2],shape)
    transformed_volumes = remap3d(moving_vol,disp)
    return transformed_volumes

def transformAtlas(args):
    x,disp = args
    shape = [tf.shape(x)[0], *K.int_shape(x)[1:4], 1]
    moving_vol = tf.reshape(x[:,:,:,:,0],shape)
    transformed_atlas = remap3d(moving_vol,disp)
    return transformed_atlas


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
    z_mean = tf.stack([pred_y[:,:,:,:,0],pred_y[:,:,:,:,1],pred_y[:,:,:,:,2]],4)
    print("Sample loss: "+str(K.int_shape(z_mean)))
    z_log_sigma = tf.stack([pred_y[:,:,:,:,3],pred_y[:,:,:,:,4],pred_y[:,:,:,:,5]],4)
    return - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)


def create_model(config):
    input_shape = (*config['resolution'][0:3],3)

    # input [0]: atlas
    # input [1]: moving
    # input [2]: fixed
    x_all = Input(shape=input_shape)
    print("0-0: "+str(K.int_shape(x_all)))

    x_1 = Lambda(lambda args:tf.stack([args[:,:,:,:,0],args[:,:,:,:,1]],axis=-1))(x_all)
    x_2 = Lambda(lambda args:tf.stack([args[:,:,:,:,0],args[:,:,:,:,2]],axis=-1))(x_all)
    print("0-1: "+str(K.int_shape(x_1)))
    print("0-2: "+str(K.int_shape(x_2)))
    vnet = build_vnet_model([16,32,32],config)
    vout_1 = Lambda(lambda args:vnet.call(args))(x_1)
    vout_2 = Lambda(lambda args:vnet.call(args))(x_2)
    print("1: "+str(K.int_shape(vout_1)))
    print("2: "+str(K.int_shape(vout_2)))
    mu_sigma_conv = build_sampling_model(config,channels_of_input=32)
    mu_1,sigma_1 = Lambda(lambda args:mu_sigma_conv.call(args))(vout_1)
    mu_2,sigma_2 = Lambda(lambda args:mu_sigma_conv.call(args))(vout_2)
    print("3-0: "+str(K.int_shape(mu_1)))
    print("3-1: "+str(K.int_shape(sigma_1)))
    print("4-0: "+str(K.int_shape(mu_2)))
    print("4-1: "+str(K.int_shape(sigma_2)))

    velo_1 = Lambda(sampling,name="VELO_1")([mu_1,sigma_1])
    velo_2 = Lambda(sampling,name="VELO_2")([mu_2,sigma_2])
    print("5: "+str(K.int_shape(velo_1)))
    print("6: "+str(K.int_shape(velo_2)))

    z_1 = Concatenate(name='KL_1')([mu_1,sigma_1])
    z_2 = Concatenate(name='KL_2')([mu_2,sigma_2])
    print("7: "+str(K.int_shape(z_1)))
    print("8: "+str(K.int_shape(z_2)))

    if config['half_res']:
        disp_low_1 = Lambda(toDisplacements(steps=config['exponentialSteps']))(velo_1)
        disp_low_2 = Lambda(toDisplacements(steps=config['exponentialSteps']))(velo_2)
        print("10: "+str(K.int_shape(disp_low_1)))
        print("11: "+str(K.int_shape(disp_low_2)))

        # upsample displacement map
        disp_upsampled_1 = Lambda(toUpscaleResampled)(disp_low_1)
        disp_upsampled_2 = Lambda(toUpscaleResampled)(disp_low_2)
        print("12: "+str(K.int_shape(disp_upsampled_1)))
        print("13: "+str(K.int_shape(disp_upsampled_2)))
        # we need to fix displacement vectors which are too small after upsampling
        disp_1 = Lambda(lambda dispMap: tf.scalar_mul(2.,dispMap),name="disp1")(disp_upsampled_1)
        disp_2 = Lambda(lambda dispMap: tf.scalar_mul(2.,dispMap),name="disp2")(disp_upsampled_2)
        print("14: "+str(K.int_shape(disp_1)))
        print("15: "+str(K.int_shape(disp_2)))
    else:
        disp_1 = Lambda(toDisplacements,name="disp1")(velo_1)
        disp_2 = Lambda(toDisplacements,name="disp2")(velo_2)

    warped_1 = Lambda(transformVolume1,name="warpToAtlas_1")([x_all,disp_1])
    warped_2 = Lambda(transformVolume2,name="warpToAtlas_2")([x_all,disp_2])
    print("16: "+str(K.int_shape(warped_1)))
    print("17: "+str(K.int_shape(warped_2)))


    invDisp_1 = Lambda(invertDisplacements)(disp_1)
    invDisp_2 = Lambda(invertDisplacements)(disp_2)
    print("18: "+str(K.int_shape(invDisp_1)))
    print("19: "+str(K.int_shape(invDisp_2)))

    warpedAtlas_1 = Lambda(transformAtlas,name="warpInv_1")([x_all,invDisp_1])
    warpedAtlas_2 = Lambda(transformAtlas,name="warpInv_2")([x_all,invDisp_2])
    print("20: "+str(K.int_shape(warpedAtlas_1)))
    print("21: "+str(K.int_shape(invDisp_2)))

    v1_to_v2_disp = Lambda(concatenateTransforms)([disp_1,invDisp_2])
    warped_1_to_2 = Lambda(transformVolume1,name="warp_v1_to_v2")([x_all,v1_to_v2_disp])
    print("22: "+str(K.int_shape(v1_to_v2_disp)))
    print("23: "+str(K.int_shape(warped_1_to_2)))

    loss = [empty_loss,
            cc3D(),
            sampleLoss,sampleLoss,
            smoothness(config['batchsize']),smoothness(config['batchsize']),
            cc3D(),cc3D(),
            cc3D(),cc3D(),
    ]
    outputs = [
        v1_to_v2_disp,
        warped_1_to_2,
        z_1,z_2,
        velo_1,velo_2,
        warped_1,warped_2,
        warpedAtlas_1,warpedAtlas_2,
    ]
    lossWeights = [# displacement
                   0.,
                   # data term / CC
                   1.0,
                   # loglikelihood
                   0.2,0.2,
                   # smoothness
                   0.000002,0.000002,
                   # data term / CC to atlas warp
                   1.0,0.8,
                   # data term / CC from atlas to vols (inv warp)
                   0.8,1.0,
                   ]
    model = Model(inputs=x_all,outputs=outputs)
    model.compile(optimizer=Adam(lr=1e-4),loss=loss,loss_weights=lossWeights,metrics=['accuracy'])
    return model
