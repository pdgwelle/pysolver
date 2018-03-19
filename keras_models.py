import pickle
with open("models/1000.pkl", "rb") as f:
    model = pickle.load(f)

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np
import matplotlib.pyplot as plt

def get_model(layers=1, layer_size=288, reg_lambda=0.01):
    dense_layers = [Dense(layer_size, input_shape=(288,), kernel_regularizer=keras.regularizers.l2(reg_lambda)), Activation('elu')]
    dense_layers.extend((layers-1) * [Dense(layer_size, input_shape=(layer_size,), kernel_regularizer=keras.regularizers.l2(reg_lambda)), Activation('elu')])
    dense_layers.append(Dense(1))
    dense_layers.append(Activation('elu'))
    k_model = Sequential(dense_layers)
    k_model.compile(optimizer="adam", loss="mean_squared_error", metrics=[keras.metrics.mean_squared_error])
    return k_model

def get_data(model):
    labels = model.training_data['moves_left']
    data = model.training_data.iloc[:, 1:]
    return labels, data

def train_model(k_model, data, labels, verbose=1, epochs=32, batch_size=128):
    history = k_model.fit(data, labels, epochs=epochs, batch_size=batch_size, verbose=verbose)
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

## Get data
labels, data = get_data(model)

## Create test data
test_model = ps.Model()
test_model.create_training_data(n_games=100, n_moves=15)
val_labels, val_data = get_data(test_model)

########################
# BASE MODEL
########################

# Train basic model, val_mse=2.76 - Not training enough epochs. Overfitting
k_model = get_model()
train_mse, val_mse, data_sizes = learning_curve(k_model, data, labels, val_data, val_labels)
plot_learning_curve(train_mse, val_mse, data_sizes)

# Increase epochs, decrease batch size, val_mse=2.85. Enough epochs. Overfitting
k_model = get_model()
train_mse, val_mse, data_sizes = learning_curve(k_model, data, labels, val_data, val_labels, 
    epochs=90, batch_size=32)
plot_learning_curve(train_mse, val_mse, data_sizes)

# Less epochs. val_mse=3.13. Not enough epochs. Badly overfitting still.
k_model = get_model()
train_mse, val_mse, data_sizes = learning_curve(k_model, data, labels, val_data, val_labels, 
    epochs=60, batch_size=32)
plot_learning_curve(train_mse, val_mse, data_sizes)

# Regularize. val_mse = 3.85. Way too much regularization!
k_model = get_model(reg_lambda = 100)
train_mse, val_mse, data_sizes = learning_curve(k_model, data, labels, val_data, val_labels, 
    epochs=90, batch_size=32)
plot_learning_curve(train_mse, val_mse, data_sizes)

# Regularize. train_mse = 2.42, val_mse = 3.12
k_model = get_model(reg_lambda = 1.0)
train_mse, val_mse, data_sizes = learning_curve(k_model, data, labels, val_data, val_labels, 
    epochs=90, batch_size=32)
plot_learning_curve(train_mse, val_mse, data_sizes)

# Try lambdas
lambdas=[0.01, 0.05, 0.1, 0.5, 1.0]
train_mse = []
val_mse = []
for lamb in lambdas:
    k_model = get_model(reg_lambda = lamb)
    train_score, val_score = get_train_validation_score(k_model, data, labels, 
        val_data, val_labels, epochs=90, batch_size=32)
    train_mse.append(train_score)
    val_mse.append(val_score)
# train_mses
# [0.47521212517392747,
#  1.595971611070737,
#  1.9369514780626318,
#  2.4808224738293485,
#  2.530707689498591]
# val_mses [2.7283635, 3.12584, 2.77948, 3.1779163, 3.3210151]

# Try simpler model
lambdas=[0.001, 0.01, 0.05, 0.1, 0.5]
train_mse = []
val_mse = []
for lamb in lambdas:
    k_model = get_model(layer_size=100, layers=1, reg_lambda = lamb)
    train_score, val_score = get_train_validation_score(k_model, data, labels, 
        val_data, val_labels, epochs=90, batch_size=32)
    train_mse.append(train_score)
    val_mse.append(val_score)
# val_mse = [2.7591174, 2.6775143, 3.0139153, 2.8819208, 3.1034288]
# train_mses
# [0.08667999209488486,
#  0.4461361649855376,
#  1.5934562222543798,
#  1.9963456792249659,
#  2.4970782410864736]

# Try more complex model
lambdas=[0.01, 0.05, 0.1, 0.5, 1.0]
train_mse = []
val_mse = []
for lamb in lambdas:
    k_model = get_model(layer_size=288*2, layers=1, reg_lambda = lamb)
    train_score, val_score = get_train_validation_score(k_model, data, labels, 
        val_data, val_labels, epochs=90, batch_size=32)
    train_mse.append(train_score)
    val_mse.append(val_score)


### Conclusions:
# Validation and training data far away. HIGH VARIANCE
# More data, less features. 
# Tried regularization. 
# Smaller model gets similar mse score


#######################
# 
#######################



# Evaluate loss
def eval_loss():
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.ylim(0,4)
    plt.show()
