import json
import os
import pickle
import random
import re

import numpy as np
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import tensorflow as tf

from keras import Input, Model, backend as K
from keras.activations import softmax
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Concatenate, Layer
from keras_preprocessing.sequence import pad_sequences


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


def setConfig(file_name):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))
    unknowns = ["gak paham", "kurang ngerti", "I don't know"]
    path = file_name + "/"
    return factory, stemmer, punct_re_escape, unknowns, file_name, path


def load_config(path, config_path):
    data = {}
    if os.path.exists(path + config_path):
        with open(path + config_path) as json_file:
            data = json.load(json_file)
    return data


def load_tokenizer(path, tokenizer_path):
    tokenizer = None
    if os.path.exists(path + tokenizer_path):
        with open(path + tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
    return tokenizer


def setParams(path, slang_path, config_path, tokenizer_path):
    list_indonesia_slang = pd.read_csv(path + slang_path, header=None).to_numpy()
    config = load_config(path, config_path)
    VOCAB_SIZE = config['VOCAB_SIZE']
    maxlen_questions = config['maxlen_questions']
    maxlen_answers = config['maxlen_answers']
    tokenizer = load_tokenizer(path, tokenizer_path)
    return list_indonesia_slang, VOCAB_SIZE, maxlen_questions, maxlen_answers, tokenizer


def check_normal_word(word_input):
    slang_result = dynamic_switcher(data_slang, word_input)
    if slang_result:
        return slang_result
    return word_input


def normalize_sentence(sentence):
    sentence = punct_re_escape.sub('', sentence.lower())
    sentence = sentence.replace('iiteung', '').replace('\n', '')
    sentence = sentence.replace('iteung', '')
    sentence = sentence.replace('teung', '')
    sentence = re.sub(r'((wk)+(w?)+(k?)+)+', '', sentence)
    sentence = re.sub(r'((xi)+(x?)+(i?)+)+', '', sentence)
    sentence = re.sub(r'((h(a|i|e)h)((a|i|e)?)+(h?)+((a|i|e)?)+)+', '', sentence)
    sentence = ' '.join(sentence.split())
    if sentence:
        sentence = sentence.strip().split(" ")
        normal_sentence = " "
        for word in sentence:
            normalize_word = check_normal_word(word)
            root_sentence = stemmer.stem(normalize_word)
            normal_sentence += root_sentence + " "
        return punct_re_escape.sub('', normal_sentence)
    return sentence


def str_to_tokens(sentence, tokenizer, maxlen_questions):
    words = sentence.lower().split()
    tokens_list = list()
    for current_word in words:
        result = tokenizer.word_index.get(current_word, '')
        if result != '':
            tokens_list.append(result)
    return pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')


def setEncoderDecoder(VOCAB_SIZE):
    enc_inputs = Input(shape=(None,))
    enc_embedding = Embedding(VOCAB_SIZE, 200, mask_zero=True)(enc_inputs)
    enc_outputs, state_h, state_c = LSTM(200, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(enc_embedding)

    # state_h = Concatenate()([forward_h, backward_h])
    # state_c = Concatenate()([forward_c, backward_c])

    enc_states = [state_h, state_c]

    dec_inputs = Input(shape=(None,))
    dec_embedding = Embedding(VOCAB_SIZE, 200, mask_zero=True)(dec_inputs)
    dec_lstm = LSTM(200, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)

    dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)

    attn_layer = AttentionLayer()
    attn_op, attn_state = attn_layer([enc_outputs, dec_outputs])
    decoder_concat_input = Concatenate(axis=-1)([dec_outputs, attn_op])

    dec_dense = Dense(VOCAB_SIZE, activation=softmax)
    output = dec_dense(decoder_concat_input)

    return dec_lstm, dec_embedding, dec_dense, dec_inputs, enc_inputs, enc_states, enc_outputs, output


def make_inference_models(dec_lstm, dec_embedding, dec_dense, dec_inputs, enc_inputs, enc_states, encoder_outputs):
    dec_state_input_h = Input(shape=(200,))
    dec_state_input_c = Input(shape=(200,))
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]

    dec_outputs, state_h, state_c = dec_lstm(dec_embedding, initial_state=dec_states_inputs)
    dec_states = [state_h, state_c]

    # dec_outputs = dec_dense(dec_outputs)

    dec_model = Model(inputs=[dec_inputs] + dec_states_inputs, outputs=[dec_outputs] + dec_states)
    enc_model = Model(inputs=enc_inputs, outputs=[encoder_outputs, enc_states])

    return enc_model, dec_model


def setModel(enc_inputs, dec_inputs, output, dec_lstm, dec_embedding, dec_dense, enc_states, encoder_outputs, file_path):
    model = Model([enc_inputs, dec_inputs], output)
    model.load_weights(file_path)

    enc_model, dec_model = make_inference_models(dec_lstm, dec_embedding, dec_dense, dec_inputs, enc_inputs, enc_states, encoder_outputs)
    return model, enc_model, dec_model


def chat(input_value, tokenizer, maxlen_answers, enc_model, dec_model):
    input_value = stemmer.stem(
        normalize_sentence(normalize_sentence(input_value))
    )

    enc_op, states_values = enc_model.predict(
        str_to_tokens(input_value, tokenizer, maxlen_questions)
    )

    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index['start']

    stop_condition = False
    decoded_translation = ''
    status = "false"

    while not stop_condition:
        dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)

        attn_layer = AttentionLayer()

        attn_op, attn_state = attn_layer([enc_op, dec_outputs])
        decoder_concat_input = Concatenate(axis=-1)([dec_outputs, attn_op])
        decoder_concat_input = dec_dense(decoder_concat_input)

        sampled_word_index = np.argmax(decoder_concat_input[0, -1, :])
        # if dec_outputs[0, -1, sampled_word_index] < 0.1:
        #     decoded_translation = unknowns[random.randint(0, (len(unknowns) - 1))]
        #     break
        sampled_word = tokenizer.word_index.get(sampled_word_index, '') + ' '
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                if word != 'end':
                    decoded_translation += ' {}'.format(word)
                sampled_word = word

        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]
        status = "true"

    return decoded_translation.strip(), str(status).lower()


factory, stemmer, punct_re_escape, unknowns, file_name, path = setConfig("lstm-bahdanau")

list_indonesia_slang, VOCAB_SIZE, maxlen_questions, maxlen_answers, tokenizer = setParams(path,
                                                                                          'daftar-slang-bahasa-indonesia.csv',
                                                                                          'config.json',
                                                                                          'tokenizer.pickle')

dec_lstm, dec_embedding, dec_dense, dec_inputs, enc_inputs, enc_states, encoder_outputs, output = setEncoderDecoder(VOCAB_SIZE)

model, enc_model, dec_model = setModel(enc_inputs,
                                       dec_inputs,
                                       output,
                                       dec_lstm,
                                       dec_embedding,
                                       dec_dense,
                                       enc_states,
                                       encoder_outputs,
                                       path + 'model-' + file_name + '.h5')

data_slang = {}
for key, value in list_indonesia_slang:
    data_slang[key] = value


def dynamic_switcher(dict_data, key):
    return dict_data.get(key, None)


def botReply(message):
    return chat(message, tokenizer, maxlen_answers, enc_model, dec_model)


while True:
    message = input("Kamu: ")
    return_message, status = botReply(message)
    print(f"ITeung: {return_message}")
