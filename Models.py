from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional,  GaussianNoise, SimpleRNN, GRU, Lambda, LSTM)
from keras.activations import relu
from train_utils import ctc_lambda_func, clipped_relu

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# //Custom
def Deep_LSTM_model(input_dim, units, activation, output_dim=29):
    """ Build a Deep LSTM based recurrent network for speech recognition task
    """
     # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    model = Sequential()
    # Batch normalize the input
    model.add(BatchNormalization(axis=-1, input_shape=(None, 161), name='BN_1'))
    
    # 1D Convs
    model.add(Conv1D(512, 5, strides=1, activation=clipped_relu, name='Conv1D_1'))
    model.add(Conv1D(512, 5, strides=1, activation=clipped_relu, name='Conv1D_2'))
    model.add(Conv1D(512, 5, strides=2, activation=clipped_relu, name='Conv1D_3'))
    
    # Batch Normalization
    model.add(BatchNormalization(axis=-1, name='BN_2'))
    
    # BiRNNs
    model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_1'), merge_mode='sum'))
    model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_2'), merge_mode='sum'))
    model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_3'), merge_mode='sum'))
    model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_4'), merge_mode='sum'))
    model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_5'), merge_mode='sum'))
    model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_6'), merge_mode='sum'))
    model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_7'), merge_mode='sum'))
    
    # Batch Normalization
    model.add(BatchNormalization(axis=-1, name='BN_3'))
    
    # FC
    model.add(TimeDistributed(Dense(1024, activation=clipped_relu, name='FC1')))
    model.add(TimeDistributed(Dense(29, activation='softmax', name='y_pred')))
    # Specify the model
    y_pred = model.outputs[0]
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def graves(input_dim=26, rnn_size=512, output_dim=29, std=0.6):
    """ Implementation of Graves 2006 model
    Architecture:
        Gaussian Noise on input
        BiDirectional LSTM
    Reference:
        ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
    """

    K.set_learning_phase(1)
    input_data = Input(name='the_input', shape=(None, input_dim))
    x = BatchNormalization(axis=-1)(input_data)

    x = GaussianNoise(std)(x)
    x = Bidirectional(LSTM(rnn_size,
                      return_sequences=True,
                      implementation=0))(x)
    x = TimeDistributed(Dense(output_dim, activation='softmax'))(x)

    y_pred = Activation('softmax', name='softmax')(x)
    
    model = Model(inputs=input_data, outputs=y_pred)
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = ...
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = ...
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = ...
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = ...
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    ...
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = ...
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = ...
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = ...
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# def final_model():
#     """ Build a deep network for speech 
#     """
#     # Main acoustic input
#     input_data = Input(name='the_input', shape=(None, input_dim))
#     # TODO: Specify the layers in your network
#     ...
#     # TODO: Add softmax activation layer
#     y_pred = ...
#     # Specify the model
#     model = Model(inputs=input_data, outputs=y_pred)
#     # TODO: Specify model.output_length
#     model.output_length = ...
#     print(model.summary())
#     return model