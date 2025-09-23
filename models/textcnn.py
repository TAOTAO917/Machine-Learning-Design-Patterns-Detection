# models/textcnn.py
import tensorflow as tf
from tensorflow.keras import layers, regularizers

@tf.keras.utils.register_keras_serializable(package="Custom")
class ReverseTemporal(layers.Layer):
    """按时间维(轴=1)反转序列，输出形状与输入一致；可序列化。"""
    def call(self, inputs):
        return tf.reverse(inputs, axis=[1])
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        return {}

def build_textcnn(seq_len:int,
                  vocab_size:int,
                  num_classes:int,
                  embedding_size:int=50,
                  filter_sizes=(2,3,4,5),
                  num_filters:int=64,
                  dropout:float=0.5,
                  l2_reg:float=0.001) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(seq_len,), dtype="int32")
    embedding = layers.Embedding(input_dim=vocab_size,
                                 output_dim=embedding_size,
                                 name="embedding")(inputs)

    conv_outs = []
    # 用可序列化层替代 Lambda
    reversed_emb = ReverseTemporal(name="reverse")(embedding)

    for k in filter_sizes:
        conv_f = layers.Conv1D(num_filters, k, activation="relu",
                               kernel_regularizer=regularizers.l2(l2_reg))(embedding)
        pool_f = layers.GlobalMaxPooling1D()(conv_f)

        conv_b = layers.Conv1D(num_filters, k, activation="relu",
                               kernel_regularizer=regularizers.l2(l2_reg))(reversed_emb)
        pool_b = layers.GlobalMaxPooling1D()(conv_b)

        conv_outs.append(layers.Concatenate()([pool_f, pool_b]))

    merged = layers.Concatenate()(conv_outs) if len(conv_outs) > 1 else conv_outs[0]
    dropped = layers.Dropout(dropout)(merged)
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=regularizers.l2(l2_reg))(dropped)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="TextCNN")
    return model
