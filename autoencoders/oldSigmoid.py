import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

data = pd.read_excel('thrombosisIPA.xlsx')

def custom_to_numeric(x):
    if isinstance(x, str) and re.match(r'^-?\d+(\.\d+)?\*$', x):
        return float(re.sub(r'\*$', '', x))
    return pd.to_numeric(x, errors='coerce')

columns_of_interest = data.iloc[:, 8:40]
columns_of_interest = columns_of_interest.applymap(custom_to_numeric)

for col in columns_of_interest.columns:
    if columns_of_interest[col].var() == 0:
        columns_of_interest.drop(columns=col, inplace=True)

#print("Mean before normalization:", columns_of_interest.mean())
#print("Standard deviation before normalization:", columns_of_interest.std())

print("Variance of each column before normalization:")
print(columns_of_interest.var())

minA = columns_of_interest.min()
maxA = columns_of_interest.max()
data_normalized = (columns_of_interest - minA)/(maxA - minA)
data_normalized.fillna(data_normalized.rolling(window=6, min_periods=1, axis=0).mean(), inplace=True)

#scaler = MinMaxScaler()
#data_normalized = scaler.fit_transform(columns_of_interest)
#data_normalized = (columns_of_interest - columns_of_interest.mean()) / columns_of_interest.std()

print("Mean after normalization:", data_normalized.mean())
print("Standard deviation after normalization:", data_normalized.std())

data_array = data_normalized#.values

# params
input_size = 32
hidden_size = 2
output_size = input_size

print("NaN values in original data:", data.isna().sum().sum())

X = tf.compat.v1.placeholder(tf.float32, shape=[None, input_size])

tf.set_random_seed(4)

weights = {
    'encoder': tf.Variable(tf.random_normal([input_size, hidden_size])),
    'decoder': tf.Variable(tf.random_normal([hidden_size, output_size]))
}
biases = {
    'encoder': tf.Variable(tf.random_normal([hidden_size])),
    'decoder': tf.Variable(tf.random_normal([output_size]))
}

encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder']), biases['encoder']))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, weights['decoder']), biases['decoder']))

y_pred = decoder

y_true = X

loss = tf.reduce_mean(tf.square(y_true - y_pred)) 
optimizer = tf.train.AdamOptimizer(learning_rate=0.08).minimize(loss) #learning rate at 600 epochs: .15

encoder = tf.keras.layers.BatchNormalization()(encoder)
decoder = tf.keras.layers.BatchNormalization()(decoder)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    num_epochs = 500
    for epoch in range(num_epochs):
        _, l = sess.run([optimizer, loss], feed_dict={X: data_array})
        if epoch % 100 == 0:
            print('Epoch {}/{}: Loss {:.4f}'.format(epoch, num_epochs, l))
    
    encoded_data, decoded_data = sess.run([encoder, decoder], feed_dict={X: data_array})

encoded_data_x = encoded_data[:, 0]
encoded_data_y = encoded_data[:, 1]

labels = data.iloc[:, -1].values

scatter = plt.scatter(encoded_data_x, encoded_data_y, c=labels, cmap='viridis')
plt.colorbar(scatter, label='Labels')
plt.xlabel('Encoded Dimension 1')
plt.ylabel('Encoded Dimension 2')
plt.title('Encoded Data Scatter Plot with Labels')

for label in np.unique(labels):
    points = encoded_data[labels == label]
    hull = ConvexHull(points)
    polygon = Polygon(points[hull.vertices], closed=False, color=scatter.to_rgba(label), alpha=0.1)
    plt.gca().add_patch(polygon)

plt.show()