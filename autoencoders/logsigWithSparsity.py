import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import re

tf.disable_v2_behavior()

from sklearn.metrics import pairwise_distances

tf.set_random_seed(17)

filename = 'thrombosisIPA.xlsx'
data = pd.read_excel(filename)

def custom_to_numeric(x):
    if isinstance(x, str) and re.match(r'^-?\d+(\.\d+)?\*$', x):
        return pd.to_numeric(re.sub(r'\*$', '', x))
    return pd.to_numeric(x, errors='coerce')

data = data.applymap(custom_to_numeric)
data.fillna(data.rolling(window=6, min_periods=1, axis=0).mean(), inplace=True)

for col in data.columns:
    if data[col].var() == 0:
        data.drop(columns=col, inplace=True)

minA = data.iloc[:, 8:40].min()
maxA = data.iloc[:, 8:40].max()
allDataN = (data.iloc[:, 8:40] - minA) / (maxA - minA)

allDataN = allDataN.values

hidden_size = 2
learning_rate = .2
training_epochs = 200
batch_size = 8
display_step = 100
l2_lambda = 0.004
sparsity_target = 0.15
sparsity_weight = 1
clip_norm = 2.5
n_neighbors = 7

X = tf.placeholder("float", [None, allDataN.shape[1]])

initializer = tf.initializers.he_normal()

weights = {
    'encoder': tf.Variable(initializer([allDataN.shape[1], hidden_size])),
    'decoder': tf.Variable(initializer([hidden_size, allDataN.shape[1]]))
}
biases = {
    'encoder': tf.Variable(tf.random_normal([hidden_size])),
    'decoder': tf.Variable(tf.random_normal([allDataN.shape[1]]))
}

encoder = tf.matmul(X, weights['encoder']) + biases['encoder']
encoder = tf.keras.layers.BatchNormalization()(encoder)
encoder = tf.nn.softplus(encoder)

decoder = tf.matmul(encoder, weights['decoder']) + biases['decoder']
decoder = tf.keras.layers.BatchNormalization()(decoder)
decoder = tf.nn.softplus(decoder)

rho_hat = tf.reduce_mean(encoder, axis=0)
sparsity_loss = tf.reduce_sum(sparsity_target * tf.log(sparsity_target / rho_hat) + (1 - sparsity_target) * tf.log((1 - sparsity_target) / (1 - rho_hat)))

l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda

loss = tf.reduce_mean(tf.square(X - decoder)) + sparsity_weight * sparsity_loss + l2_loss

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

grads_and_vars = optimizer.compute_gradients(loss)
grads, variables = zip(*grads_and_vars)

clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm)

train_op = optimizer.apply_gradients(zip(clipped_grads, variables))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(len(allDataN) /   batch_size)
    for epoch in range(training_epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x = allDataN[i * batch_size: (i + 1) * batch_size]
            _, c, gradients = sess.run([train_op, loss, grads_and_vars], feed_dict={X: batch_x})
            avg_cost += c / total_batch
        
        gradient_norm = np.sqrt(np.sum([np.sum(grad**2) for grad, _ in gradients]))
        if gradient_norm >= 1 and epoch > 525:
            print("Minimum gradient threshold reached. Stopping training. Epoch:", '%04d' % (epoch))
            break
        
        if epoch % display_step == 0:
            print("Gradient: ", '{:.9f}'.format(gradient_norm))
            print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(avg_cost))

    encoded_data = sess.run(encoder, feed_dict={X: allDataN})
    
    encoder_weights = sess.run(weights['encoder'])

tsne = TSNE(n_components=2, perplexity=5, learning_rate=50, n_iter=1000)
tsne_data = tsne.fit_transform(encoded_data)

outcome_column = data.columns[-1]

#plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=data[outcome_column], cmap='coolwarm')
#plt.title('2D t-SNE Embedding')
#plt.xlabel('TSNE Component 1')
#plt.ylabel('TSNE Component 2')
#plt.colorbar(label='Outcome')
#plt.show()

plt.figure(figsize=(8, 6))
scatter = plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=data.iloc[:, -1], cmap='coolwarm')
plt.title('Encoded Data Scatterplot')
plt.xlabel('Encoded Feature 1')
plt.ylabel('Encoded Feature 2')
plt.colorbar(scatter, label='Outcome')

average_abs_weights = np.mean(np.abs(encoder_weights), axis=1)

sorted_indices = np.argsort(average_abs_weights)[::-1]
sorted_weights = average_abs_weights[sorted_indices]
sorted_features = data.columns[8:40][sorted_indices]

plt.figure(figsize=(12, 8))
bars = plt.bar(range(allDataN.shape[1]), sorted_weights, align='center', alpha=0.8)
plt.xticks(range(allDataN.shape[1]), sorted_features, rotation=90)
plt.title('Average Absolute Encoder Weights Across Hidden Dimensions (Sorted)')
plt.xlabel('Input Features')
plt.ylabel('Average Absolute Weight')
plt.tight_layout()
plt.show()

#for i, txt in enumerate(data.iloc[:, -1]):
#    plt.annotate(int(data.iloc[:, 43][i]), (encoded_data[i, 0], encoded_data[i, 1]))

#plt.show()

y = data.iloc[:, -1].values
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(encoded_data, y)

predictions = knn.predict(encoded_data)

accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)