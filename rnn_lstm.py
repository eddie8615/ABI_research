import numpy as np
import os
import json
import pandas as pd
import fnmatch
import argparse

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras.backend as K
from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Input, Dense, Masking, LSTM, Dropout, TimeDistributed, Bidirectional, Embedding, Concatenate
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from keras.models import save_model

from numpy.random import seed
from numpy import array

from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


sections = ['train/', 'validation/', 'test/']

min_time = 1.0
n_features = 25
random_seed = 42

epochs = 20
batch_size = 4
embedding_dim = 300
dropout = 0.3
learning_rate = 0.0001

dimensions = ['Arousal', 'Valence', 'Dominance']
tuning = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    mode = 'latefusion'

    global sentence_level
    global data_path
    global train_length
    global val_length
    global test_length
    global seq_len

    if args.sentence == 'y' or args.sentence == 'yes':
        sentence_level = True
        data_path = './LLDs_podcast/'
        train_length = 2684
        val_length = 726
        test_length = 1498

    elif args.sentence == 'n' or args.sentence == 'no':
        sentence_level = False
        data_path = './LLDs_conversation/'
        train_length = 96
        val_length = 30
        test_length = 44

    else:
        print('Invalid argument')
        return

    if args.model == 'latefusion':
        mode = 'latefusion'
    elif args.model == 'hierarchy':
        mode = 'hierarchy'
    else:
        print('Invalid argument')
        return

    # initialise
    set_random_seed(random_seed)

    if sentence_level:
        seq_len = 388

        segment_path = os.path.relpath('./MSP Data/Time Labels/segments.json')
        f = open(segment_path, 'r')
        timing_data = json.load(f)
        train_transcript, val_transcript, test_transcript = sentence_transcripts(timing_data)
    else:
        train_transcript, val_transcript, test_transcript = conversation_transcripts()
        seq_len = 847


    train_cleaned = train_transcript['txt'].apply(clean_text)
    val_cleaned = val_transcript['txt'].apply(clean_text)
    test_cleaned = test_transcript['txt'].apply(clean_text)
    train_val = pd.concat([train_cleaned, val_cleaned], ignore_index=True)
    combined_cleaned = pd.concat([train_cleaned, val_cleaned, test_cleaned], ignore_index=True)

    print("Cleaned train text: %d" % (len(train_cleaned)))
    print("Cleaned validation text: %d" % (len(val_cleaned)))
    print("Cleaned test text: %d" % (len(test_cleaned)))
    print("Cleaned all text: %d" % (len(combined_cleaned)))

    x_train, x_val, x_test, y_train, y_val, y_test = load_data()

    x_final = np.concatenate([x_train, x_val, x_test])
    x_scaler = get_scaler(x_final)
    # Debugging
    x_test_scaled = scale_data(x_scaler, x_test)
    x_final_scaled = scale_data(x_scaler, x_final)

#     need to appropriate transformation for multi-task learning
    y_train_dta = transform_mtl(y_train)
    y_val_dta = transform_mtl(y_val)
    y_test_dta = transform_mtl(y_test)
    y_final = []
    for i in range(len(dimensions)):
        y_final.append(np.concatenate([y_train_dta[i], y_val_dta[i], y_test_dta[i]]))

    print('End of data preparation...')

    epoch = 1
    if mode == 'latefusion':
        model = create_late_fusion_model(combined_cleaned, n_units=512, optimizer='rmsprop', use_glove=True)
    else:
        model = create_hierarchical_model(combined_cleaned, n_units=512, optimizer='rmsprop', use_glove=True)

    min_loss = float('inf')
    early_stop = 5
    while epoch <= epochs:
        if early_stop < 0:
            break
        history = model.fit([combined_cleaned, x_final_scaled], y_final,
                    batch_size=batch_size,
                    shuffle=True,
                    initial_epoch=epoch-1,
                    epochs=epoch,
                    validation_split=0.01)

        val_loss = history.history['val_loss'][-1]
        if val_loss < min_loss:
            early_stop = 5
            min_loss = val_loss
            arousal_ccc = history.history['val_Arousal_ccc'][-1]
            valence_ccc = history.history['val_Valence_ccc'][-1]
            dominance_ccc = history.history['val_Dominance_ccc'][-1]
            print('Arousal CCC: %.4f' % arousal_ccc)
            print('Valence CCC: %.4f' % valence_ccc)
            print('Dominance CCC: %.4f' % dominance_ccc)
            model_name = './checkpoints/lstm-' + mode
            model.save_weights(model_name, save_format='tf')
        else:
            early_stop -= 1

        epoch += 1



def create_late_fusion_model(combined_cleaned, n_units=256, optimizer='rmsprop', use_glove=True):
    embedding_layer, text_vec = import_glove(combined_cleaned)
    vocab_size = calc_vocab_size(combined_cleaned)

    ling_model = Sequential()
    ling_inputs = Input(shape=(1,), dtype=tf.string)
    vec = text_vec(ling_inputs)
    if use_glove:
        embed = embedding_layer(vec)
    else:
        embed = Embedding(vocab_size + 1,
                          embedding_dim,
                          input_length=seq_len,
                          trainable=True,
                          name='Embedding')(vec)

    lstm1 = LSTM(n_units, return_sequences=True)(embed)
    lstm2 = LSTM(n_units, return_sequences=True, dropout=dropout)(lstm1)
    lstm2 = LSTM(n_units, return_sequences=True, dropout=dropout)(lstm2)
    lstm2 = Dropout(dropout)(lstm2)
    ling_output = [TimeDistributed(Dense(1), name='ling_' + name)(lstm2) for i, name in enumerate(dimensions)]
    ling_model = Model(inputs=ling_inputs, outputs=ling_output)

    acoustic_model = Sequential()
    inputs = Input(shape=(seq_len, n_features * 2), dtype=float)
    mask = Masking()(inputs)
    lstm_1 = LSTM(n_units, return_sequences=True)(mask)
    lstm_2 = LSTM(n_units, return_sequences=True, dropout=dropout)(lstm_1)
    lstm_2 = LSTM(n_units, return_sequences=True, dropout=dropout)(lstm_2)
    lstm_2 = Dropout(dropout)(lstm_2)
    acous_output = [TimeDistributed(Dense(1), name='acoustic_' + name)(lstm_2) for i, name in enumerate(dimensions)]

    acoustic_model = Model(inputs=inputs, outputs=acous_output)
    late_fusion = []
    for output in ling_model.output:
        late_fusion.append(output)
    for output in acoustic_model.output:
        late_fusion.append(output)
    concat = Concatenate()(late_fusion)
    lstm_last = LSTM(128, return_sequences=True)(concat)
    #     dense_last = TimeDistributed(Dense(32))(concat)
    output = [TimeDistributed(Dense(1), name=name)(lstm_last) for i, name in enumerate(dimensions)]

    bimodal_model = Model(inputs=[ling_inputs, inputs], outputs=output)

    if optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    else:
        opt = SGD(learning_rate=learning_rate)
    bimodal_model.compile(optimizer=opt, loss=ccc_loss, metrics=[ccc])
    return bimodal_model


def create_hierarchical_model(combined_cleaned, n_units=256, optimizer='rmsprop', use_glove=True):
    embedding_layer, text_vec = import_glove(combined_cleaned)
    vocab_size = calc_vocab_size(combined_cleaned)

    acoustic_input = Input(shape=(seq_len, n_features * 2), name='acoustic_input')
    masking = Masking(name='acoustic_mask')(acoustic_input)

    linguistic_input = Input(shape=(1,), dtype=tf.string)
    vec = text_vec(linguistic_input)
    if use_glove:
        embed = embedding_layer(vec)
    else:
        embed = Embedding(vocab_size + 1,
                          embedding_dim,
                          input_length=seq_len,
                          trainable=True,
                          name='Embedding')(vec)

    linguistic_lstm = LSTM(n_units, return_sequences=True, name='linguistic_lstm1')(embed)
    linguistic_lstm = LSTM(n_units, return_sequences=True, name='linguistic_lstm2')(linguistic_lstm)
    linguistic_lstm = LSTM(n_units, return_sequences=True, name='linguistic_lstm3')(linguistic_lstm)

    concat = Concatenate()([masking, linguistic_lstm])

    bi_lstm = LSTM(n_units, return_sequences=True, dropout=dropout, name='bi_lstm1')(concat)
    bi_lstm = LSTM(n_units, return_sequences=True, dropout=dropout, name='bi_lstm2')(bi_lstm)
    bi_lstm = LSTM(n_units, return_sequences=True, dropout=dropout, name='bi_lstm3')(bi_lstm)
    output = [TimeDistributed(Dense(1), name=name)(bi_lstm) for i, name in enumerate(dimensions)]
    bimodal = Model([linguistic_input, acoustic_input], output)

    if optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    else:
        opt = SGD(learning_rate=learning_rate)

    bimodal.compile(optimizer=opt, loss=ccc_loss, metrics=[ccc])
    return bimodal


def import_glove(combined_cleaned):
    vocab_size = calc_vocab_size(combined_cleaned)
    text_vec = text_vectorize(combined_cleaned, vocab_size)
    embeddings_index = dict()
    f = open('embeddings/glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
    for i, word in enumerate(text_vec.get_vocabulary()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(vocab_size + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=seq_len,
                                trainable=True,
                                name='GloVe')
    return embedding_layer, text_vec


def text_vectorize(combined_cleaned, vocab_size):
    sentence_len = [len(sent.split()) for sent in combined_cleaned.tolist()]
    sent_len = max(sentence_len)
    print('Max sentence length: %d' % (sent_len))
    text_vec = TextVectorization(max_tokens=vocab_size,
                                 pad_to_max_tokens=True,
                                 output_sequence_length=seq_len,
                                 output_mode='int')
    text_vec.adapt(combined_cleaned)

    return text_vec


def calc_vocab_size(combined_cleaned):
    vocab = set()

    for i in range(len(combined_cleaned)):
        text = combined_cleaned[i]
        tokens = text.split()
        for token in tokens:
            vocab.add(token)
    vocab_size = len(vocab)

    return vocab_size


def clean_text(text):
    punctuation = [i for i in ',./\\;:\'@#~[{]}=+-_)(*&^%$£"!`)]']
    STOPWORDS = set(stopwords.words('english'))

    text = text.replace("'s", "")
    text = "".join([" " if t in punctuation else t for t in text]).lower()
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwords from text
    return text


def transform_mtl(y):
    y_dta = []
    index = [0,1,2]
    for i in index:
        dim = np.empty((len(y), y.shape[1], 1))
        dim[:,:,0] = y[:,:,i]
        y_dta.append(dim)
    return y_dta


def sentence_transcripts(timing_data):
    for i, section in enumerate(sections):
        files = fnmatch.filter(os.listdir(data_path + section), '*.txt')
        files.sort()
        text = []
        filename = []
        for file in files:
            inst = file.split('.')[0]

            if 'MSP-PODCAST_0153' in inst or 'MSP-PODCAST_1188_0020' in inst:
                continue
            start = timing_data[inst]['Start_Time']
            end = timing_data[inst]['End_Time']
            if end - start < min_time:
                continue
            filename.append(inst)
            with open(data_path + section + file) as f:
                lines = f.readlines()
            if len(lines) == 0:
                text.append('')
            else:
                text.append(lines[0])
        if i == 0:
            df_train = pd.DataFrame({'Filename': filename, 'txt': text})

        elif i == 1:
            df_val = pd.DataFrame({'Filename': filename, 'txt': text})

        else:
            df_test = pd.DataFrame({'Filename': filename, 'txt': text})

    return df_train, df_val, df_test


def conversation_transcripts():
    transcripts_path = data_path + 'transcripts/'
    train = []
    val = []
    test = []
    for i, section in enumerate(sections):
        files = fnmatch.filter(os.listdir(data_path + section), '*.csv')
        insts = [file.split('.')[0] for file in files]
        if i == 0:
            train = insts
        elif i == 1:
            val = insts
        else:
            test = insts

    transcripts = fnmatch.filter(os.listdir(transcripts_path), '*.txt')
    transcripts.sort()
    train_files = []
    val_files = []
    test_files = []

    train_text = []
    val_text = []
    test_text = []
    for transcript in transcripts:

        with open(transcripts_path + transcript) as f:
            lines = f.readlines()
        inst = transcript.split('.')[0]
        if inst in train:
            train_files.append(inst)
            train_text.append(lines[0])
        elif inst in val:
            val_files.append(inst)
            val_text.append(lines[0])
        else:
            test_files.append(inst)
            test_text.append(lines[0])
    df_train = pd.DataFrame({'Filename': train_files, 'txt': train_text})
    df_val = pd.DataFrame({'Filename': val_files, 'txt': val_text})
    df_test = pd.DataFrame({'Filename': test_files, 'txt': test_text})

    return df_train, df_val, df_test


def load_data():
    '''
    Loading raw acoustic data
    '''

    # load all data to get a scaler that covers all data
    print("Loading training samples...")
    x_train = load_features(data_path + 'train.txt', skip_header=False, skip_instname=False)
    train_labels = load_labels(data_path + 'train_labels_lag_compensated.txt', skip_header=False, skip_instname=False)
    y_train = train_labels.reshape((train_length, seq_len, 3))

    print("Loading finished, Scaling...")

    # Scaling acoustic features
    # Scaling labels from [-100,100] to [-1, 1]
    f = lambda x: x * 0.01
    y_train_scaled = f(y_train)
    x_train = x_train.reshape((train_length, seq_len, n_features * 2))
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train_scaled.shape)
    print("End of loading and preprocessing training samples")

    print("Loading validation samples...")
    x_validation = load_features(data_path + 'validation.txt', skip_header=False, skip_instname=False)
    val_labels = load_labels(data_path + 'validation_labels_lag_compensated.txt', skip_header=False,
                             skip_instname=False)
    y_validation = val_labels.reshape((val_length, seq_len, 3))
    print("Loading finished, Scaling...")
    x_val = x_validation.reshape((val_length, seq_len, n_features * 2))
    y_val_scaled = f(y_validation)

    print('x_validation shape:', x_val.shape)
    print('y_validation shape:', y_val_scaled.shape)
    print("End of loading and preprocessing validation samples")

    print("Loading testing samples...")
    x_test = load_features(data_path + 'test.txt', skip_header=False, skip_instname=False)
    test_labels = load_labels(data_path + 'test_labels_lag_compensated.txt', skip_header=False, skip_instname=False)
    y_test = test_labels.reshape((test_length, seq_len, 3))
    print("Loading finished, Scaling...")

    y_test_scaled = f(y_test)
    x_test = x_test.reshape((test_length, seq_len, n_features * 2))

    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test_scaled.shape)
    print("End of loading and preprocessing test samples")

    return x_train, x_val, x_test, y_train_scaled, y_val_scaled, y_test_scaled


def load_features(filename,
                  skip_header=True,
                  skip_instname=True,
                  delim=' ',
                  num_lines=0):
    if num_lines == 0:
        num_lines = get_num_lines(filename, skip_header)

    data = np.empty(
        (num_lines, n_features * 2), float)

    with open(filename, 'r') as csv_file:
        if skip_header:
            next(csv_file)
        c = 0
        for line in tqdm(csv_file):
            offset = 0
            if skip_instname:
                offset = line.find(delim) + 1
            data[c, :] = np.fromstring(line[offset:], dtype=float, sep=delim)
            c += 1

    return data


def get_num_lines(filename, skip_header):
    with open(filename, 'r') as csv_file:
        if skip_header:
            next(csv_file)
        c = 0
        for line in csv_file:
            c += 1
    return c


def load_labels(filename,
                  skip_header=True,
                  skip_instname=True,
                  delim=' ',
                  num_lines=0):
    if num_lines == 0:
        num_lines = get_num_lines(filename, skip_header)

    data = np.empty(
        (num_lines, 3), float)

    with open(filename, 'r') as csv_file:
        if skip_header:
            next(csv_file)
        c = 0
        for line in tqdm(csv_file):
            offset = 0
            if skip_instname:
                offset = line.find(delim) + 1
            data[c, :] = np.fromstring(line[offset:], dtype=float, sep=delim)
            c += 1

    return data


def get_scaler(x):
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[2])
    x_scaler = StandardScaler()
    x_scaler.fit(x)

    return x_scaler


def scale_data(scaler, data):
    shape = data.shape
    if data.ndim > 2:
        data = data.reshape(-1, data.shape[2])
    scaled = scaler.transform(data)
    scaled = scaled.reshape((shape[0], shape[1], shape[2]))

    return scaled


def ccc(gold, pred):
    gold       = K.squeeze(gold, axis=-1)
    pred       = K.squeeze(pred, axis=-1)
    gold_mean  = K.mean(gold, axis=-1, keepdims=True)
    pred_mean  = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var   = K.mean(K.square(gold-gold_mean), axis=-1,  keepdims=True)
    pred_var   = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.epsilon())
    return ccc


def ccc_loss(gold, pred):
    ccc_loss = K.constant(1.) - ccc(gold, pred)
    return ccc_loss



if __name__ == '__main__':
    main()
