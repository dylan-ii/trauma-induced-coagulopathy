import tensorflow as tf
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from sklearn.decomposition import PCA

def custom_to_numeric(x):
    if isinstance(x, str) and re.match(r'^-?\d+(\.\d+)?\*$', x):
        return float(re.sub(r'\*$', '', x))
    return pd.to_numeric(x, errors='coerce')

xls = pd.ExcelFile('AllData.xlsx')

data_biomarkers = pd.read_excel(xls, sheet_name='Blood Biomarkers')
data_proteomics = pd.read_excel(xls, sheet_name='Combination')

data_biomarkers_transposed = data_biomarkers.transpose()
labels = data_biomarkers_transposed.iloc[1:, -1].values

data_proteomics_transposed = data_proteomics.transpose()

proteomicsFrame = data_proteomics_transposed.iloc[1:62, 1:26]
#biomarkersFrame = data_biomarkers_transposed.iloc[1:64, 8:40]

proteomics_features = proteomicsFrame#pd.concat([proteomicsFrame, biomarkersFrame], axis=0)

print(proteomics_features.shape)

columns_of_interest = proteomics_features.applymap(custom_to_numeric)

columns_to_drop = []
a=0

for col in columns_of_interest.columns:
    print(col)
    print(columns_of_interest[col].var())
    print(columns_of_interest[col].shape)
    #if columns_of_interest[col].shape != [63,]:
    #    columns_to_drop.append(col)
    #    a+=1
    #    continue
    if columns_of_interest[col].var() < 0.001:
        a+=1
        columns_to_drop.append(col)

print(a)

columns_of_interest.drop(columns=columns_to_drop, inplace=True)

print("Variance of each column before normalization:")
print(columns_of_interest.var())

minA = columns_of_interest.min()
maxA = columns_of_interest.max()
data_normalized = (columns_of_interest - minA) / (maxA - minA)
data_normalized.fillna(data_normalized.rolling(window=6, min_periods=1, axis=0).mean(), inplace=True)

print("Mean after normalization:", data_normalized.mean())
print("Standard deviation after normalization:", data_normalized.std())

data_array = data_normalized.values

print(data_array.shape)

input_size = data_array.shape[1]
hidden_size = 2

encoder_inputs = tf.keras.layers.Input(shape=(input_size,))
encoder_regularized = tf.keras.layers.Dense(hidden_size, activation='tanh',
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(encoder_inputs)
encoder_dropout = tf.keras.layers.Dropout(0.2)(encoder_regularized)
encoder_model = tf.keras.models.Model(inputs=encoder_inputs, outputs=encoder_dropout)

decoder_inputs = tf.keras.layers.Input(shape=(hidden_size,))
decoder_regularized = tf.keras.layers.Dense(input_size, activation='tanh',
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(decoder_inputs)
decoder_dropout = tf.keras.layers.Dropout(0.2)(decoder_regularized)
decoder_model = tf.keras.models.Model(inputs=decoder_inputs, outputs=decoder_dropout)

autoencoder_outputs = decoder_model(encoder_model(encoder_inputs))
autoencoder = tf.keras.models.Model(inputs=encoder_inputs, outputs=autoencoder_outputs)

autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')

num_epochs = 400
history = autoencoder.fit(data_array, data_array, epochs=num_epochs, verbose=1)

encoded_data = encoder_model.predict(data_array)

# pca = PCA(n_components=2)
# encoded_data_pca = pca.fit_transform(encoded_data)

colors = ['red' if label == 1 else 'black' for label in labels]

plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=colors, s=26)
plt.xlabel('Encoded Dimension 1', fontsize=18, labelpad=20)
plt.ylabel('Encoded Dimension 2', fontsize=18, labelpad=20)
plt.title('Encoded Data as a Scatter Plot with Labels', fontsize=22)

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='1', markerfacecolor='red', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='0', markerfacecolor='black', markersize=10)
]
plt.legend(handles=legend_elements, title='Mortality', loc='best')

plt.show()