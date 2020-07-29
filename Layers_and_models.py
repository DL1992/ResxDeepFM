import tensorflow as tf
from deepctr.models import *
from deepctr.feature_column import *
from deepctr.layers.core import *
from deepctr.layers.interaction import *
from deepctr.layers.utils import *
from tensorflow.python.keras.layers import *


class ResDNN(DNN):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **skip_rate**: int. the frequency of skip connections
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, skip_rate=1,
                 **kwargs):
        self.skip_rate = skip_rate
        super(ResDNN, self).__init__(hidden_units, activation, l2_reg, dropout_rate, use_bn, seed, **kwargs)

    def build(self, input_shape):
        self.conv_layers = [Conv2D(self.hidden_units[i], kernel_size=(1, 1), padding='valid') for i in
                            range(len(self.hidden_units))]

        super(ResDNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs
        shortcut = deep_input

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            # fc = Dense(self.hidden_size[i], activation=None, \
            #           kernel_initializer=glorot_normal(seed=self.seed), \
            #           kernel_regularizer=l2(self.l2_reg))(deep_input)
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            if i % self.skip_rate == 0:
                shortcut = tf.expand_dims(shortcut, 1)
                shortcut = tf.expand_dims(shortcut, 1)
                shortcut = self.conv_layers[i](shortcut, training=training)
                shortcut = tf.squeeze(shortcut, [1, 2])
                fc = add_func([fc, shortcut])
                shortcut = fc
            deep_input = fc

        return deep_input

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate, 'seed': self.seed,
                  'skip_rate': self.skip_rate}
        base_config = super(ResDNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ResCIN(CIN):
    """Compressed Interaction Network used in xDeepFM.This implemention is
    adapted from code that the author of the paper published on https://github.com/Leavingseason/xDeepFM.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)`` ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .
      Arguments
        - **layer_size** : list of int.Feature maps in each layer.
        - **activation** : activation function used on feature maps.
        - **split_half** : bool.if set to False, half of the feature maps in each hidden will connect to output unit.
        - **seed** : A Python integer to use as random seed.
      References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, layer_size=(128, 128), activation='relu', split_half=True, l2_reg=1e-5, seed=1024, skip_rate=1,
                 **kwargs):
        self.skip_rate = skip_rate
        super(ResCIN, self).__init__(layer_size, activation, split_half, l2_reg, seed, **kwargs ** kwargs)

    def build(self, input_shape):
        self.conv_layers = [Conv2D(self.layer_size[i], kernel_size=(1, 1), padding='valid') for i in
                            range(len(self.layer_size))]

        super(ResDNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        dim = int(inputs.get_shape()[-1])
        hidden_nn_layers = [inputs]
        final_result = []

        split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
        for idx, layer_size in enumerate(self.layer_size):
            split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)

            dot_result_m = tf.matmul(
                split_tensor0, split_tensor, transpose_b=True)

            dot_result_o = tf.reshape(
                dot_result_m, shape=[dim, -1, self.field_nums[0] * self.field_nums[idx]])

            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            curr_out = tf.nn.conv1d(
                dot_result, filters=self.filters[idx], stride=1, padding='VALID')

            curr_out = tf.nn.bias_add(curr_out, self.bias[idx])

            curr_out = self.activation_layers[idx](curr_out)

            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            if self.split_half:
                if idx != len(self.layer_size) - 1:
                    next_hidden, direct_connect = tf.split(
                        curr_out, 2 * [layer_size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)
        result = reduce_sum(result, -1, keep_dims=False)

        return result

    def compute_output_shape(self, input_shape):
        if self.split_half:
            featuremap_num = sum(
                self.layer_size[:-1]) // 2 + self.layer_size[-1]
        else:
            featuremap_num = sum(self.layer_size)
        return (None, featuremap_num)

    def get_config(self, ):

        config = {'layer_size': self.layer_size, 'split_half': self.split_half, 'activation': self.activation,
                  'seed': self.seed, 'skip_rate': self.skip_rate}
        base_config = super(ResCIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def ResxDeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 256),
               cin_layer_size=(128, 128,), cin_split_half=True, cin_activation='relu', l2_reg_linear=0.00001,
               l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, seed=1024, dnn_dropout=0,
               dnn_activation='relu', dnn_use_bn=False, task='binary', skip_rate=1):
    """Instantiates the ResxDeepFM architecture.
    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param cin_layer_size: list,list of positive integer or empty list, the feature maps  in each hidden layer of Compressed Interaction Network
    :param cin_split_half: bool.if set to True, half of the feature maps in each hidden will connect to output unit
    :param cin_activation: activation function used on feature maps
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: L2 regularizer strength applied to deep net
    :param l2_reg_cin: L2 regularizer strength applied to CIN.
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param skip_rate: int, the frequency of skip connections
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    fm_input = concat_func(sparse_embedding_list, axis=1)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_output = ResDNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                        dnn_use_bn, seed, skip_rate)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(dnn_output)

    final_logit = add_func([linear_logit, dnn_logit])

    if len(cin_layer_size) > 0:
        exFM_out = ResCIN(cin_layer_size, cin_activation,
                          cin_split_half, l2_reg_cin, seed, skip_rate)(fm_input)
        exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)
        final_logit = add_func([final_logit, exFM_logit])

    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model


def FGCNN_ResxDeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 256),
                     cin_layer_size=(128, 128,), cin_split_half=True, cin_activation='relu', l2_reg_linear=0.00001,
                     l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, seed=1024, dnn_dropout=0,
                     conv_kernel_width=(10, 10, 7, 7), conv_filters=(14, 16, 18, 20),
                     new_maps=(3, 3, 3, 3),
                     pooling_width=(1, 1, 1, 1),
                     dnn_activation='relu', dnn_use_bn=False, task='binary', skip_rate=1):
    """Instantiates the FGCNN_ResxDeepFM architecture.
    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param cin_layer_size: list,list of positive integer or empty list, the feature maps  in each hidden layer of Compressed Interaction Network
    :param cin_split_half: bool.if set to True, half of the feature maps in each hidden will connect to output unit
    :param cin_activation: activation function used on feature maps
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: L2 regularizer strength applied to deep net
    :param l2_reg_cin: L2 regularizer strength applied to CIN.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
     :param skip_rate: int, the frequency of skip connections
    :return: A Keras model instance.
    """

    features = build_input_features(linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    # sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
    #                                                                      l2_reg_embedding, init_std, seed)

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    ##############
    deep_emb_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, seed)
    fg_deep_emb_list, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, seed,
                                                     prefix='fg')

    fg_input = concat_func(fg_deep_emb_list, axis=1)
    origin_input = concat_func(deep_emb_list, axis=1)

    if len(conv_filters) > 0:
        new_features = FGCNNLayer(
            conv_filters, conv_kernel_width, new_maps, pooling_width)(fg_input)
        combined_input = concat_func([origin_input, new_features], axis=1)
    else:
        combined_input = origin_input
    ############

    fm_input = combined_input

    dnn_input = combined_dnn_input(deep_emb_list, dense_value_list)
    dnn_input = concat_func([dnn_input, Flatten()(combined_input)])
    dnn_output = ResDNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                        dnn_use_bn, seed, skip_rate)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(dnn_output)

    final_logit = add_func([linear_logit, dnn_logit])

    if len(cin_layer_size) > 0:
        exFM_out = ResCIN(cin_layer_size, cin_activation,
                          cin_split_half, l2_reg_cin, seed)(fm_input)
        exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)
        final_logit = add_func([final_logit, exFM_logit])

    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model


def FGCNN_xDeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 256),
                  cin_layer_size=(128, 128,), cin_split_half=True, cin_activation='relu', l2_reg_linear=0.00001,
                  l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, init_std=0.0001, seed=1024, dnn_dropout=0,
                  conv_kernel_width=(10, 10, 7, 7), conv_filters=(14, 16, 18, 20),
                  new_maps=(3, 3, 3, 3),
                  pooling_width=(1, 1, 1, 1),
                  dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the FGCNN_xDeepFM architecture.
    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param cin_layer_size: list,list of positive integer or empty list, the feature maps  in each hidden layer of Compressed Interaction Network
    :param cin_split_half: bool.if set to True, half of the feature maps in each hidden will connect to output unit
    :param cin_activation: activation function used on feature maps
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: L2 regularizer strength applied to deep net
    :param l2_reg_cin: L2 regularizer strength applied to CIN.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    # sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
    #                                                                      l2_reg_embedding, init_std, seed)

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    ##############
    deep_emb_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, seed)
    fg_deep_emb_list, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, seed,
                                                     prefix='fg')

    fg_input = concat_func(fg_deep_emb_list, axis=1)
    origin_input = concat_func(deep_emb_list, axis=1)

    if len(conv_filters) > 0:
        new_features = FGCNNLayer(
            conv_filters, conv_kernel_width, new_maps, pooling_width)(fg_input)
        combined_input = concat_func([origin_input, new_features], axis=1)
    else:
        combined_input = origin_input
    ############

    fm_input = combined_input

    dnn_input = combined_dnn_input(deep_emb_list, dense_value_list)
    dnn_input = concat_func([dnn_input, Flatten()(combined_input)])
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     dnn_use_bn, seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(dnn_output)

    final_logit = add_func([linear_logit, dnn_logit])

    if len(cin_layer_size) > 0:
        exFM_out = CIN(cin_layer_size, cin_activation,
                       cin_split_half, l2_reg_cin, seed)(fm_input)
        exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)
        final_logit = add_func([final_logit, exFM_logit])

    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
