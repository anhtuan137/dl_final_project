import tensorflow as tf
from tensorflow import keras


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE) cho Keras.
    Tính trên giá trị đã scale (MinMaxScaler).
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def r2(y_true, y_pred):
    """
    Coefficient of Determination (R²) cho Keras.
    Cẩn thận: tính theo mini-batch, không phải toàn bộ dataset,
    nhưng đủ dùng để theo dõi xu hướng theo epoch.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))

    return 1.0 - ss_res / (ss_tot + 1e-8)
