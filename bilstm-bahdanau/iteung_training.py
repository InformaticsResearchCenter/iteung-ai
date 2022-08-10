# -*- coding: utf-8 -*-
import json
import os
import pickle

import pandas as pd
import tensorflow as tf
from keras import Input, Model, backend as K
from keras.activations import softmax
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Concatenate, Layer
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state
            inputs: (batchsize * 1 * de_in_dim)
            states: (batchsize * 1 * de_latent_dim)
            """

            assert_msg = "States must be an iterable. Got {} of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch size * en_seq_len * latent_dim
            W_a_dot_s = K.dot(encoder_out_seq, self.W_a)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>', U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)
            if verbose:
                print('Ws+Uh>', Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """

            assert_msg = "States must be an iterable. Got {} of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        fake_state_c = K.sum(encoder_out_seq, axis=1)
        fake_state_e = K.sum(encoder_out_seq, axis=2)  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]

path = "bilstm-bahdanau/"
try:
    os.makedirs(path)
except:
    pass

dataset = pd.read_csv('clean_qa.txt', delimiter="\t", header=None)

questions_train = dataset.iloc[:, 0].values.tolist()
answers_train = dataset.iloc[:, 1].values.tolist()

questions_test = dataset.iloc[:, 0].values.tolist()
answers_test = dataset.iloc[:, 1].values.tolist()


def save_tokenizer(tokenizer):
    with open('bilstm-bahdanau/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_config(key, value):
    data = {}
    if os.path.exists(path + 'config.json'):
        with open(path + 'config.json') as json_file:
            data = json.load(json_file)

    data[key] = value
    with open(path + 'config.json', 'w') as outfile:
        json.dump(data, outfile)


target_regex = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'0123456789'
tokenizer = Tokenizer(filters=target_regex, lower=True)
tokenizer.fit_on_texts(questions_train + answers_train + questions_test + answers_test)
save_tokenizer(tokenizer)

VOCAB_SIZE = len(tokenizer.word_index) + 1
save_config('VOCAB_SIZE', VOCAB_SIZE)
print('Vocabulary size : {}'.format(VOCAB_SIZE))

tokenized_questions_train = tokenizer.texts_to_sequences(questions_train)
maxlen_questions_train = max([len(x) for x in tokenized_questions_train])
save_config('maxlen_questions', maxlen_questions_train)
encoder_input_data_train = pad_sequences(tokenized_questions_train, maxlen=maxlen_questions_train, padding='post')

print(encoder_input_data_train.shape)

tokenized_questions_test = tokenizer.texts_to_sequences(questions_test)
maxlen_questions_test = max([len(x) for x in tokenized_questions_test])
save_config('maxlen_questions', maxlen_questions_test)
encoder_input_data_test = pad_sequences(tokenized_questions_test, maxlen=maxlen_questions_test, padding='post')

print(encoder_input_data_test.shape)

tokenized_answers_train = tokenizer.texts_to_sequences(answers_train)
maxlen_answers_train = max([len(x) for x in tokenized_answers_train])
save_config('maxlen_answers', maxlen_answers_train)
decoder_input_data_train = pad_sequences(tokenized_answers_train, maxlen=maxlen_answers_train, padding='post')

print(decoder_input_data_train.shape)

tokenized_answers_test = tokenizer.texts_to_sequences(answers_test)
maxlen_answers_test = max([len(x) for x in tokenized_answers_test])
save_config('maxlen_answers', maxlen_answers_test)
decoder_input_data_test = pad_sequences(tokenized_answers_test, maxlen=maxlen_answers_test, padding='post')

print(decoder_input_data_test.shape)

for i in range(len(tokenized_answers_train)):
    tokenized_answers_train[i] = tokenized_answers_train[i][1:]
padded_answers_train = pad_sequences(tokenized_answers_train, maxlen=maxlen_answers_train, padding='post')
decoder_output_data_train = to_categorical(padded_answers_train, num_classes=VOCAB_SIZE)

print(decoder_output_data_train.shape)

for i in range(len(tokenized_answers_test)):
    tokenized_answers_test[i] = tokenized_answers_test[i][1:]
padded_answers_test = pad_sequences(tokenized_answers_test, maxlen=maxlen_answers_test, padding='post')
decoder_output_data_test = to_categorical(padded_answers_test, num_classes=VOCAB_SIZE)

print(decoder_output_data_test.shape)

enc_inp = Input(shape=(None,))
enc_embedding = Embedding(VOCAB_SIZE, 200, mask_zero=True)(enc_inp)
enc_outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(200, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(enc_embedding)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
enc_states = [state_h, state_c]

dec_inp = Input(shape=(None,))
dec_embedding = Embedding(VOCAB_SIZE, 200, mask_zero=True)(dec_inp)
dec_lstm = LSTM(200 * 2, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)
dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)

attn_layer = AttentionLayer()
attn_op, attn_state = attn_layer([enc_outputs, dec_outputs])
decoder_concat_input = Concatenate(axis=-1)([dec_outputs, attn_op])

dec_dense = Dense(VOCAB_SIZE, activation=softmax)
output = dec_dense(decoder_concat_input)

logdir = os.path.join(path, "logs")
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

checkpoint = ModelCheckpoint(os.path.join(path, 'model-{epoch:02d}-{loss:.2f}.hdf5'),
                             monitor='loss',
                             verbose=1,
                             save_best_only=True, mode='auto', period=100)

model = Model([enc_inp, dec_inp], output)
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

batch_size = 64
epochs = 400
model.fit([encoder_input_data_train, decoder_input_data_train],
          decoder_output_data_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([encoder_input_data_test, decoder_input_data_test], decoder_output_data_test),
          callbacks=[tensorboard_callback, checkpoint])
model.save(os.path.join(path, 'model-' + path.replace("/", "") + '.h5'))
