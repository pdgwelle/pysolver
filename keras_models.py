import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np
import matplotlib.pyplot as plt

import pysolver as ps
from lr_finder import LRFinder

def get_model(layer_info=[288], reg_lambda=0.01):
    
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

if __name__ == '__main__':
    
    ## Read in data
    import pickle
    with open("models/1000.pkl", "rb") as f:
        model = pickle.load(f)

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

    ### Conclusions:
    # Validation and training data far away. HIGH VARIANCE
    # More data, less features. 
    # Tried regularization. 
    # Smaller model gets similar mse score


    #######################
    # 5000 games
    #######################

    ## Read in data
    import pickle
    with open("models/5000.pkl", "rb") as f:
        model = pickle.load(f)

    ## Get data
    labels, data = get_data(model)

    ## Create test data
    test_model = ps.Model()
    test_model.create_training_data(n_games=500, n_moves=15)
    val_labels, val_data = get_data(test_model)

    # Try initial model (too few epochs!) (1 min 13 secs)
    k_model = get_model(layer_info=[100], reg_lambda=0.01)
    train_mse, val_mse, data_sizes = learning_curve(k_model, data, labels, val_data, val_labels)
    plot_learning_curve(train_mse, val_mse, data_sizes)
    # train_mse
    # [1.8325009634213658,
    #  1.2401973347101631,
    #  1.1821984344654672,
    #  1.1757188645812653,
    #  1.157283694763424]
    # val_mse = [2.2390652, 1.9152844, 1.9313699, 1.8837734, 1.9307281]

    # Try model with more epochs 8min 24s. Curves look good, still high variance. Need more data. 
    # Note: Amazon instance p2.xlarge was 23min 35s. Slowwww!!
    k_model = get_model(layers=1, layer_size=100, reg_lambda=0.01)
    train_mse, val_mse, data_sizes = learning_curve(k_model, data, labels, val_data, val_labels,
        epochs=90, batch_size=32)
    plot_learning_curve(train_mse, val_mse, data_sizes)
    # train_mse
    # [0.9428296794020089,
    #  1.1503472427330395,
    #  1.269752991675186,
    #  1.3116868469545586,
    #  1.3320530513854418]
    # val_mse
    # [2.2947514, 2.1764157, 2.0113559, 1.991767, 1.9731245]

    ## Try overfitting. 18min 36s. Really similar performance!
    k_model = get_model(layers=2, layer_size=288, reg_lambda=0.01)
    train_mse, val_mse, data_sizes = learning_curve(k_model, data, labels, val_data, val_labels,
        epochs=90, batch_size=32)
    plot_learning_curve(train_mse, val_mse, data_sizes)
    # train_mse:
    # [0.8974917681554113,
    #  1.2013781601839726,
    #  1.2536018186741464,
    #  1.3177748978620816,
    #  1.3143270486459122]
    # val_mse
    # [2.3333464, 2.1525126, 2.260423, 2.0248573, 1.9488932]

    ## Try overfitting with one layer. 26min 3s. Again, similar performance.
    k_model = get_model(layers=1, layer_size=288*3, reg_lambda=0.01)
    train_mse, val_mse, data_sizes = learning_curve(k_model, data, labels, val_data, val_labels,
        epochs=90, batch_size=32)
    plot_learning_curve(train_mse, val_mse, data_sizes)
    # train_mse
    # [0.9242459848268305,
    #  1.1361261460718863,
    #  1.2795783124455025,
    #  1.3083641902805985,
    #  1.3381138709705225]
    # val_mse
    # [2.2532637, 2.1082494, 2.049546, 1.9930592, 1.9454465]

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
    # train_mse
    # [0.6348636344228105,
    #  1.3675692196862317,
    #  2.09179469127037,
    #  2.222595258844961,
    #  2.5319955567086097]
    # val_mse
    # [2.0787704, 1.9801466, 2.3308702, 2.3849738, 2.6314483]

    #################
    # 25000 games
    #################

    ## Read in data
    import pickle
    with open("models/25000.pkl", "rb") as f:
        model = pickle.load(f)

    ## Get data
    labels, data = get_data(model)

    ## Create test data
    import pickle
    with open("models/1000.pkl", "rb") as f:
        test_model = pickle.load(f)
        val_labels, val_data = get_data(test_model)


    # Base model. High variance
    k_model = get_model(layers=1, layer_size=100, reg_lambda=0.01)
    train_mse, val_mse, data_sizes = learning_curve(k_model, data, labels, val_data, val_labels,
        epochs=90, batch_size=32)
    plot_learning_curve(train_mse, val_mse, data_sizes)
    # In [6]: train_mse
    # Out[6]: 
    # [1.6119696227863909,
    #  1.6136937567233591,
    #  1.6571004809239545,
    #  1.6637332187800657,
    #  1.670475259670603]

    # In [7]: val_mse
    # Out[7]: [1.7951659, 1.7616024, 1.9366946, 1.7412255, 1.7654753]

    # Larger network. Still high bias! 1h 25min 39s
    k_model = get_model(layers=2, layer_size=288, reg_lambda=0.01)
    train_mse, val_mse, data_sizes = learning_curve(k_model, data, labels, val_data, val_labels,
        epochs=90, batch_size=32)
    plot_learning_curve(train_mse, val_mse, data_sizes)
    # In [10]: train_mse
    # Out[10]: 
    # [1.6880550534168288,
    #  1.692283646907604,
    #  1.7231736503733641,
    #  1.7340434309323651,
    #  1.7400832447578536]

    # In [11]: val_mse
    # Out[11]: [1.9112828, 1.8388073, 1.8732424, 1.8381759, 1.8142205]

    #####
    # Conclusions:
    # hard to get bias down. use less data

    model.training_data = model.training_data.sample(frac=0.5).reset_index(drop=True)
    labels, data = get_data(model)

    # 29 min. could go slightly deeper
    k_model = get_model(layer_info=[100], reg_lambda=0.01)
    train_mse, val_mse, data_sizes = learning_curve(k_model, data, labels, val_data, val_labels,
        epochs=90, batch_size=32)
    plot_learning_curve(train_mse, val_mse, data_sizes)
    # In [74]: train_mse
    # Out[74]: 
    # [1.4528788555633918,
    #  1.5811162331004198,
    #  1.6196184033497512,
    #  1.6314786876197944,
    #  1.6486933191562783]

    # In [75]: val_mse
    # Out[75]: [2.3969858, 1.9032001, 1.8346411, 1.7741485, 1.7828928]

    k_model = get_model(layer_info=[72,72], reg_lambda=0.01)
    train_mse, val_mse, data_sizes = learning_curve(k_model, data, labels, val_data, val_labels,
        epochs=90, batch_size=32)
    plot_learning_curve(train_mse, val_mse, data_sizes)

    # In [80]: train_mse
    # Out[80]: 
    # [1.4217503914193312,
    #  1.5680213167172843,
    #  1.5834921137149074,
    #  1.6074862096292608,
    #  1.6331608621657787]

    # In [81]: val_mse
    # Out[81]: [2.3058429, 1.8699716, 1.7746309, 1.773375, 1.8344156]

    k_model = get_model(layer_info=[72,72,72], reg_lambda=0.01)
    train_mse, val_mse, data_sizes = learning_curve(k_model, data, labels, val_data, val_labels,
        epochs=90, batch_size=32)
    # In [95]: train_mse
    # Out[95]: 
    # [1.5225229180278603,
    #  1.665514293773594,
    #  1.6771803459582455,
    #  1.6915359080126326,
    #  1.6867056246929004]

    # In [96]: val_mse
    # Out[96]: [2.0354822, 1.8670318, 1.9844126, 2.0279083, 1.8204767]

    k_model = get_model(layer_info=[288*3], reg_lambda=0.01)
    train_mse, val_mse, data_sizes = learning_curve(k_model, data, labels, val_data, val_labels,
        epochs=90, batch_size=32)
