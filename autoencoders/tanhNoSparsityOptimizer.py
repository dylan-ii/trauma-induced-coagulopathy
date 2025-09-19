import tensorflow as tf
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA

def custom_to_numeric(x):
    if isinstance(x, str) and re.match(r'^-?\d+(\.\d+)?\*$', x):
        return float(re.sub(r'\*$', '', x))
    return pd.to_numeric(x, errors='coerce')

xls = pd.ExcelFile('AllData.xlsx')

data_biomarkers = pd.read_excel(xls, sheet_name='Blood Biomarkers')
data_proteomics = pd.read_excel(xls, sheet_name='Proteomics')

data_biomarkers_transposed = data_biomarkers.transpose()
labels = data_biomarkers_transposed.iloc[1:, -1].values

data_proteomics_transposed = data_proteomics.transpose()

proteomics_features = data_proteomics_transposed.iloc[1:64, 1:26]

columns_of_interest = proteomics_features.applymap(custom_to_numeric)

for col in columns_of_interest.columns:
    if columns_of_interest[col].var() == 0:
        columns_of_interest.drop(columns=col, inplace=True)

print("Variance of each column before normalization:")
print(columns_of_interest.var())

minA = columns_of_interest.min()
maxA = columns_of_interest.max()
data_normalized = (columns_of_interest - minA) / (maxA - minA)
data_normalized.fillna(data_normalized.rolling(window=6, min_periods=1, axis=0).mean(), inplace=True)

print("Mean after normalization:", data_normalized.mean())
print("Standard deviation after normalization:", data_normalized.std())

data_array = data_normalized.values

input_size = data_array.shape[1]
hidden_size = 2 

def build_autoencoder(lr, l2_reg, dropout_rate):

    encoder_inputs = tf.keras.layers.Input(shape=(input_size,))
    encoder_regularized = tf.keras.layers.Dense(hidden_size, activation='tanh',
                                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(encoder_inputs)
    encoder_dropout = tf.keras.layers.Dropout(dropout_rate)(encoder_regularized)
    encoder_model = tf.keras.models.Model(inputs=encoder_inputs, outputs=encoder_dropout)

    decoder_inputs = tf.keras.layers.Input(shape=(hidden_size,))
    decoder_regularized = tf.keras.layers.Dense(input_size, activation='tanh',
                                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(decoder_inputs)
    decoder_dropout = tf.keras.layers.Dropout(dropout_rate)(decoder_regularized)
    decoder_model = tf.keras.models.Model(inputs=decoder_inputs, outputs=decoder_dropout)

    autoencoder_outputs = decoder_model(encoder_model(encoder_inputs))
    autoencoder = tf.keras.models.Model(inputs=encoder_inputs, outputs=autoencoder_outputs)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

    num_epochs = 250
    autoencoder.fit(data_array, data_array, epochs=num_epochs, verbose=0)

    encoded_data = encoder_model.predict(data_array)

    db_index = davies_bouldin_score(encoded_data, labels)

    return db_index

#grid search parameters
learning_rates = np.arange(0.001,0.02,0.001)#[0.001, 0.01, 0.05]
l2_regs = np.arange(0.01,0.10,0.01)
dropout_rates = [0.2,0.1,0]

best_db_index = 100000
best_params = {}

for lr in learning_rates:
    for l2_reg in l2_regs:
        for dropout_rate in dropout_rates:
            db_index = []
            for i in range(3):
                db_index = np.append(db_index, build_autoencoder(lr, l2_reg, dropout_rate))

                if db_index[i] == 0: db_index[i] = 100
            print(f"Parameters: lr={lr}, l2={l2_reg}, dropout={dropout_rate} --> Davies-Bouldin Index: {np.mean(db_index)}")
            
            if (np.mean(db_index) < best_db_index) and np.mean(db_index) != 0:
                best_db_index = np.mean(db_index)
                best_params = {'lr': lr, 'l2_reg': l2_reg, 'dropout_rate': dropout_rate}

print(f"\nBest parameters: {best_params} with Davies-Bouldin Index: {best_db_index}")