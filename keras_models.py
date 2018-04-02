import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

import pysolver as ps
from lr_finder import LRFinder

def get_model(layer_info=[288], reg_lambda=0.01, lr=0.01, decay=0.0):
    
    if(len(layer_info) == 0):
        print("layer_info cannot be empty")
        return None

    dense_layers = [Dense(layer_info[0], input_shape=(288,), kernel_regularizer=keras.regularizers.l2(reg_lambda)), Activation('elu')]
    
    last_layer = layer_info[0]
    for layer in layer_info:
        dense_layers.extend([Dense(layer, input_shape=(last_layer,), kernel_regularizer=keras.regularizers.l2(reg_lambda)), Activation('elu')])
        last_layer = layer

    dense_layers.append(Dense(1))
    dense_layers.append(Activation('elu'))
    k_model = Sequential(dense_layers)
    adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)
    k_model.compile(optimizer=adam, loss="mean_squared_error", metrics=[keras.metrics.mean_squared_error])
    return k_model

def get_data(model):
    labels = model.training_data['moves_left']
    data = model.training_data.iloc[:, 1:]
    return labels, data

def train_model(k_model, data, labels, val_data=None, val_labels=None, verbose=1, epochs=32, batch_size=128):
    if((val_data is None) | (val_labels is None)):
        history = k_model.fit(data, labels, epochs=epochs, batch_size=batch_size, verbose=verbose)
    else:
        history = k_model.fit(data, labels, epochs=epochs, batch_size=batch_size, verbose=verbose,
            validation_data = (val_data, val_labels))
    return history

def learning_curve(k_model, data, labels, val_data, val_labels, n_steps=5, epochs=32, batch_size=128):
    data_sizes_float = np.linspace(0, len(data), n_steps+1)[1:]
    data_sizes = [int(np.floor(size)) for size in data_sizes_float]

    train_mse = []
    val_mse = []
    for size in data_sizes:
        train_score, validation_score = get_train_validation_score(k_model, data, labels,
            val_data, val_labels, epochs, batch_size, size)
        train_mse.append(train_score)
        val_mse.append(validation_score)

    return train_mse, val_mse, data_sizes

def get_train_validation_score(k_model, data, labels, val_data, val_labels, epochs, batch_size, size=None):
    if(size is None): size=len(data)
    
    row_indices = np.random.randint(0, len(data), size)
    X = data.iloc[row_indices, :]
    y = labels[row_indices]
    k_model.reset_states()
    history = train_model(k_model, X, y, verbose=0, epochs=epochs, batch_size=batch_size)
    train_score = history.history['mean_squared_error'][-1]
    validation_score = np.mean((k_model.predict(val_data).flatten()-val_labels)**2)

    return train_score, validation_score

def plot_learning_curve(train_mse, val_mse, data_sizes):
    plt.plot(data_sizes, train_mse, label='train_mse')
    plt.plot(data_sizes, val_mse, label='val_mse')
    plt.xlabel("Number of samples")
    plt.ylabel("Mean Squared Error")
    plt.ylim(0,4)
    plt.axhline(1, color='red')
    plt.legend()
    plt.show()

def get_errors(k_model, val_data, val_labels, plot=False):
    y_hat = pd.Series(k_model.predict(val_data).flatten())
    predict_df = pd.DataFrame({'y_hat': y_hat, 'y': val_labels})

    unique_moves = np.sort(val_labels.unique())

    mae_list = []
    for move in unique_moves:
        subset = predict_df[predict_df['y'] == move]
        mae = np.sum(np.abs((subset['y'] - subset['y_hat']))) / len(subset)
        mae_list.append(mae)

    if(plot):
        plt.bar(unique_moves, mae_list)
        plt.ylabel("Mean Average Error")
        plt.xlabel("Moves Away")
        plt.show()

    return mae_list

if __name__ == '__main__':

    ## Read in data
    with open("models/1000000.pkl", "rb") as f:
        model = pickle.load(f)

    ## Get data
    labels, data = get_data(model)

    ## Create test data
    with open("models/1000.pkl", "rb") as f:
        test_model = pickle.load(f)
        val_labels, val_data = get_data(test_model)
