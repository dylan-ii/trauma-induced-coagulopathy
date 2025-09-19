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
data_proteomics = pd.read_excel(xls, sheet_name='Blood Biomarkers')

data_biomarkers_transposed = data_biomarkers.transpose()
labels = data_biomarkers_transposed.iloc[1:, -1].values

data_proteomics_transposed = data_proteomics.transpose()

proteomicsFrame = data_proteomics_transposed.iloc[1:, 8:40]
#biomarkersFrame = data_biomarkers_transposed.iloc[1:64, 8:40]

proteomics_features = proteomicsFrame#pd.concat([proteomicsFrame, biomarkersFrame], axis=0)

print(proteomics_features.shape)

columns_of_interest = proteomics_features.applymap(custom_to_numeric)

columns_to_drop = []

for col in columns_of_interest.columns:
    if columns_of_interest[col].var() < 0.001:
        columns_to_drop.append(col)

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

# parameters    
input_size = data_array.shape[1]
hidden_size = 3

encoder_inputs = tf.keras.layers.Input(shape=(input_size,))
encoder_regularized = tf.keras.layers.Dense(hidden_size, activation='tanh',
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(encoder_inputs)
encoder_dropout = tf.keras.layers.Dropout(0.0)(encoder_regularized)
encoder_model = tf.keras.models.Model(inputs=encoder_inputs, outputs=encoder_dropout)

decoder_inputs = tf.keras.layers.Input(shape=(hidden_size,))
decoder_regularized = tf.keras.layers.Dense(input_size, activation='tanh',
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(decoder_inputs)
decoder_dropout = tf.keras.layers.Dropout(0.0)(decoder_regularized)
decoder_model = tf.keras.models.Model(inputs=decoder_inputs, outputs=decoder_dropout)

autoencoder_outputs = decoder_model(encoder_model(encoder_inputs))
autoencoder = tf.keras.models.Model(inputs=encoder_inputs, outputs=autoencoder_outputs)

autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')

num_epochs = 250
history = autoencoder.fit(data_array, data_array, epochs=num_epochs, verbose=1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

encoded_data = encoder_model.predict(data_array)

colors = ['red' if label == 1 else 'black' for label in labels]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2], c=colors, s=26)

ax.set_xlabel('Encoded Dimension 1', fontsize=18, labelpad=20)
ax.set_ylabel('Encoded Dimension 2', fontsize=18, labelpad=20)
ax.set_zlabel('Encoded Dimension 3', fontsize=18, labelpad=20)
ax.set_title('Encoded Data as a 3D Scatter Plot with Labels', fontsize=22)

def plot_convex_hull(data, color, alpha=0.1, edge_alpha=0.05):
    if len(data) < 4:
        return
    
    hull = ConvexHull(data)
    poly3d = [[data[vertex] for vertex in simplex] for simplex in hull.simplices]
    
    collection = Poly3DCollection(poly3d, facecolors=color, linewidths=1, edgecolors=(0, 0, 0, edge_alpha), alpha=alpha)
    ax.add_collection3d(collection)

data_label_1 = encoded_data[np.array(labels) == 1]
data_label_0 = encoded_data[np.array(labels) == 0]

plot_convex_hull(data_label_1, color='red')
plot_convex_hull(data_label_0, color='black')

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='1', markerfacecolor='red', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='0', markerfacecolor='black', markersize=10)
]
ax.legend(handles=legend_elements, title='Mortality', loc='best')

plt.show()
