from tensorflow.python.ops import array_ops
import tensorflow as tf
import numpy as np
from keras.layers import K
from dense_3D_spatial_transformer import Dense3DSpatialTransformer


def volumeGradients(tf_vf):
    #batch_size, xaxis, yaxis, zaxis, depth = \
    shapes = tf.shape(tf_vf)
    #batch_size = tf.placeholder("float",shape=None)
    dx = tf_vf[:, 1:, :, :, :] - tf_vf[:, :-1, :, :, :]
    dy = tf_vf[:, :, 1:, :, :] - tf_vf[:, :, :-1, :, :]
    dz = tf_vf[:, :, :, 1:, :] - tf_vf[:, :, :, :-1, :]

    # Return tensors with same size as original image by concatenating
    # zeros. Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
    shape = tf.stack([shapes[0], 1, shapes[2], shapes[3], shapes[4]])
    dx = array_ops.concat([dx, tf.zeros(shape, tf_vf.dtype)], 1)
    dx = array_ops.reshape(dx, tf.shape(tf_vf))

    #shape = tf.stack([batch_size, xaxis, 1, zaxis, depth])
    shape = tf.stack([shapes[0], shapes[1], 1, shapes[3], shapes[4]])
    dy = array_ops.concat([dy, array_ops.zeros(shape, tf_vf.dtype)], 2)
    dy = array_ops.reshape(dy, tf.shape(tf_vf))

    #shape = tf.stack([batch_size, xaxis, yaxis, 1, depth])
    shape = tf.stack([shapes[0], shapes[1], shapes[2], 1, shapes[4]])
    dz = array_ops.concat([dz, array_ops.zeros(shape, tf_vf.dtype)], 3)
    dz = array_ops.reshape(dz, tf.shape(tf_vf))

    '''
    normalization = lambda unnormalized_tensor: tf.div(
        tf.subtract(
            unnormalized_tensor,
            tf.reduce_min(unnormalized_tensor)
        ),
        tf.subtract(
            tf.reduce_max(unnormalized_tensor),
            tf.reduce_min(unnormalized_tensor)
        )
    )
    
    new_shape = np.array([-1, xaxis, yaxis, zaxis, 3], dtype=np.float32)
    return tf.map_fn(normalization, tf.reshape(array_ops.stack([dx, dy, dz], 4), new_shape))
    '''
    return tf.reshape(array_ops.stack([dx, dy, dz], 4), [shapes[0], shapes[1], shapes[2], shapes[3], 3])


def tfVectorFieldExp(grad, grid):
    N = 2
    shape = tf.shape(grid)

    id_x = tf.reshape(grid[:, :, :, :, 0], [shape[0], shape[1], shape[2], shape[3], 1])
    id_y = tf.reshape(grid[:, :, :, :, 1], [shape[0], shape[1], shape[2], shape[3], 1])
    id_z = tf.reshape(grid[:, :, :, :, 2], [shape[0], shape[1], shape[2], shape[3], 1])

    ux = grad[:, :, :, :, 0]
    uy = grad[:, :, :, :, 1]
    uz = grad[:, :, :, :, 2]

    dvx = ux / (2.0 ** N)
    dvy = uy / (2.0 ** N)
    dvz = uz / (2.0 ** N)
    dvx = id_x + tf.reshape(dvx, [shape[0], shape[1], shape[2], shape[3], 1])
    dvy = id_y + tf.reshape(dvy, [shape[0], shape[1], shape[2], shape[3], 1])
    dvz = id_z + tf.reshape(dvz, [shape[0], shape[1], shape[2], shape[3], 1])

    for n in range(0, N - 1):
        cache_tf = tf.reshape(tf.identity(tf.stack([dvx - id_x, dvy - id_y, dvz - id_z], 4)),
                              [shape[0], shape[1], shape[2], shape[3], 3])

        dvx = tf.reshape(Dense3DSpatialTransformer()([dvx, cache_tf]), [shape[0], shape[1], shape[2], shape[3], 1])
        dvy = tf.reshape(Dense3DSpatialTransformer()([dvy, cache_tf]), [shape[0], shape[1], shape[2], shape[3], 1])
        dvz = tf.reshape(Dense3DSpatialTransformer()([dvz, cache_tf]), [shape[0], shape[1], shape[2], shape[3], 1])

    return tf.reshape(tf.stack([dvx - id_x, dvy - id_y, dvz - id_z], 4), [shape[0], shape[1], shape[2], shape[3], 3])