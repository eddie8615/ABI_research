import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
import keras.backend as K
from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Input, Dense, Masking, LSTM, Dropout, TimeDistributed, Bidirectional
from tensorflow.keras.optimizers import RMSprop, Adam
from keras.models import save_model

from numpy.random import seed
from tensorflow.keras.utils import set_random_seed

train_length = 25253
val_length = 9471
test_length = 13794

time_step = 399
n_features = 25
random_seed = 42

data_path = './Data/'

def main():
    set_random_seed(random_seed)

    print("Loading training samples...")
    x_train = load_features(data_path + 'train.txt', skip_header=False, skip_instname=False)
    train_labels = load_labels(data_path + 'train_labels.txt')
    y_train = train_labels.iloc[:,-3:]
    print('X_train length: %d' % (len(x_train)))
    print('y_train length: %d' % (len(y_train)))
    x_scaler, y_scaler = get_scaler(x_train, y_train)

    x_val = load_features(data_path+'validation.txt', skip_header=False, skip_instname=False)
    val_labels = load_labels(data_path + 'validation_labels.txt')
    y_val = val_labels.iloc[:,-3:]
    print('X_val length: %d' % (len(x_val)))
    print('y_val length: %d' % (len(y_val)))

    model = create_model()
    x_train = scale_data(x_scaler, x_train)
    x_val = scale_data(x_scaler, x_val)
    x_train = x_train.reshape(train_length, time_step, n_features)
    x_val = x_val.reshape(val_length, time_step, n_features)

    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))
    plot_learningCurve(history, 10)

    model.save_model('lstm.h5')


def create_model(n_units1=64, n_units2=32):
    model = Sequential()
    model.add(LSTM(n_units1, input_shape=(time_step, n_features), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(n_units2, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(3))
    rms = RMSprop(learning_rate=0.0001)
    model.compile(loss='mse', optimizer=rms)
    return model


def plot_learningCurve(history, epoch):
  # Plot training & validation accuracy values
    epoch_range = range(1, epoch+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


def load_features(filename,
                  skip_header=True,
                  skip_instname=True,
                  delim=' ',
                  num_lines=0):
    if num_lines == 0:
        num_lines = get_num_lines(filename, skip_header)

    data = np.empty(
        (num_lines, 25), float)

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


def load_batch_features(filename, start_index=0, amount=0):
    delim = ' '

    data = np.empty((amount, 25), float)

    with open(filename, 'r') as csv_file:
        for i, line in tqdm(enumerate(csv_file)):
            if i < start_index:
                continue
            if i >= start_index + amount:
                break
            index = i - start_index
            data[index, :] = np.fromstring(line, dtype=float, sep=delim)
    return data


def load_batch_labels(filename, start_index=1, amount=0):
    labels = np.empty((amount, 3), float)
    delim = ','

    with open(filename, 'r') as csv_file:
        for i, line in tqdm(enumerate(csv_file)):
            if i < start_index:
                continue
            if i >= start_index + amount:
                break
            cols = np.fromstring(line, dtype=float, sep=delim)
            index = i - start_index
            labels[index, :] = cols[1:]
    return labels


def get_num_lines(filename, skip_header):
    with open(filename, 'r') as csv_file:
        if skip_header:
            next(csv_file)
        c = 0
        for line in csv_file:
            c += 1
    return c


def load_labels(filename):
    return pd.read_csv(filename)


def get_scaler(x, y):
    x_scaler = StandardScaler()
    x_scaler.fit(x)
    y_scaler = StandardScaler()
    y_scaler.fit(y)

    return x_scaler, y_scaler


def scale_data(scaler, data):
    if data.ndim > 2:
        data = data.reshape(-1, data.shape[2])
    scaled = scaler.transform(data)

    return scaled


if __name__ == '__main__':
    main()
