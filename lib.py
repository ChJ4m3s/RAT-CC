import tensorflow.keras.backend as kB
import tensorflow as tf

def cosine_similarity_loss(y_true, y_predict):
    shape = kB.shape(y_true)
    if len(shape) == 3:
        y_true = kB.reshape(y_true, (shape[0], shape[1] * shape[2]))

    y_pred = tf.convert_to_tensor(y_predict)
    y_true = tf.cast(y_true, y_pred.dtype)

    y_true_t = tf.transpose(y_true)
    y_predict_t = tf.transpose(y_pred)

    m_true = kB.dot(y_true, y_true_t)
    m_pred = kB.dot(y_predict, y_predict_t)

    ones = tf.ones(tf.shape(y_predict)[0])

    d_true = tf.linalg.diag_part(m_true)
    d_pred = tf.linalg.diag_part(m_pred)
    d_true_t = tf.transpose(d_true)
    d_pred_t = tf.transpose(d_pred)

    h2_true = tf.tensordot(d_true_t, ones, 0)
    h2_pred = tf.tensordot(d_pred_t, ones, 0)
    h2_true_t = tf.transpose(h2_true)
    h2_pred_t = tf.transpose(h2_pred)

    dm_true = h2_true + h2_true_t - 2 * m_true
    dm_pred = h2_pred + h2_pred_t - 2 * m_pred

    max_true = tf.math.reduce_max(dm_true)
    min_true = tf.math.reduce_min(dm_true)
    dm_true_norm = tf.math.divide(tf.math.subtract(dm_true, min_true), max_true - min_true + 0.001)

    max_pred = tf.math.reduce_max(dm_pred)
    min_pred = tf.math.reduce_min(dm_pred)
    dm_pred_norm = tf.math.divide(tf.math.subtract(dm_pred, min_pred), max_pred - min_pred + 0.001)

    mae = tf.math.reduce_mean(tf.abs(dm_true_norm - dm_pred_norm), axis=-1)

    return mae


def mean_absolute_percentage_error(y_true, y_pred):
    shape = kB.shape(y_true)
    if len(shape) == 3:
        y_true = kB.reshape(y_true, (shape[0], shape[1] * shape[2]))
    shape = kB.shape(y_pred)
    if len(shape) == 3:
        y_pred = kB.reshape(y_pred, (shape[0], shape[1] * shape[2]))
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return kB.mean(tf.abs(y_pred - y_true), axis=-1)


def repeat(x):
    stepMatrix = kB.ones_like(x[0][:, :, :1])  # matrix with ones, shaped as (batch, steps, 1)
    latentMatrix = kB.expand_dims(x[1], axis=1)  # latent vars, shaped as (batch, 1, latent_dim)

    return kB.batch_dot(stepMatrix, latentMatrix)