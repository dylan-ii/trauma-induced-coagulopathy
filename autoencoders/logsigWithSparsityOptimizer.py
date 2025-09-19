import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import re

tf.disable_v2_behavior()

from sklearn.metrics import pairwise_distances

def davies_bouldin_index(encoded_data, y):
    unique_classes = np.unique(y)
    centroids = []
    intra_cluster_distances = []
    
    for cls in unique_classes:
        centroid = np.mean(encoded_data[y == cls], axis=0)
        centroids.append(centroid)
        
        intra_dist = np.mean(pairwise_distances(encoded_data[y == cls], centroid.reshape(1, -1)))
        intra_cluster_distances.append(intra_dist)
    
    centroids = np.array(centroids)
    intra_cluster_distances = np.array(intra_cluster_distances)
    
    db_index = 0.0
    for i in range(len(unique_classes)):
        for j in range(len(unique_classes)):
            if i != j:
                inter_cluster_distance = np.linalg.norm(centroids[i] - centroids[j])
                
                db_index += (intra_cluster_distances[i] + intra_cluster_distances[j]) / inter_cluster_distance
    
    db_index /= len(unique_classes)
    
    return db_index

filename = 'thrombosisIPA.xlsx'
data = pd.read_excel(filename)

def custom_to_numeric(x):
    if isinstance(x, str) and re.match(r'^-?\d+(\.\d+)?\*$', x):
        return float(re.sub(r'\*$', '', x))
    return pd.to_numeric(x, errors='coerce')

data = data.applymap(custom_to_numeric)

for col in data.columns:
    if data[col].var() == 0:
        data.drop(columns=col, inplace=True)

minA = data.iloc[:, 8:40].min()
maxA = data.iloc[:, 8:40].max()
allDataN = (data.iloc[:, 8:40] - minA) / (maxA - minA)

allDataN.fillna(allDataN.rolling(window=6, min_periods=1, axis=0).mean(), inplace=True)

allDataN = allDataN.values

# hyperparameters to test
batch_sizes = [8]#, 16, 32]
learning_rates = [.15]
n_neighbors_values = [11]

results = {}

# sparsity parameters to test
sparsity_targets = [.15,.3,.5]#np.arange(0.1,1,0.1)
sparsity_weights = np.arange(0,6,1) 
l2_lambdas = [.002,.004]#np.arange(0.002,0.01,0.002)

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        for n_neighbors in n_neighbors_values:
            for sparsity_target in sparsity_targets:
                for sparsity_weight in sparsity_weights:
                    for l2_lambda in l2_lambdas:
                        print("Testing combination: Batch Size =", batch_size, "Learning Rate =", learning_rate, "n_neighbors =", n_neighbors, "Sparsity Target =", sparsity_target, "Sparsity Weight =", sparsity_weight)
                        
                        accuracies = []
                        for _ in range(1):
                            hidden_size = 2
                            training_epochs = 550
                            display_step = 250

                            X = tf.placeholder("float", [None, allDataN.shape[1]])

                            weights = {
                                'encoder': tf.Variable(tf.random_normal([allDataN.shape[1], hidden_size])),
                                'decoder': tf.Variable(tf.random_normal([hidden_size, allDataN.shape[1]]))
                            }
                            biases = {
                                'encoder': tf.Variable(tf.random_normal([hidden_size])),
                                'decoder': tf.Variable(tf.random_normal([allDataN.shape[1]]))
                            }

                            encoder = tf.nn.softplus(tf.add(tf.matmul(X, weights['encoder']), biases['encoder']))

                            decoder = tf.nn.softplus(tf.add(tf.matmul(encoder, weights['decoder']), biases['decoder']))

                            rho_hat = tf.reduce_mean(encoder, axis=0)
                            sparsity_loss = tf.reduce_sum(sparsity_target * tf.log(sparsity_target / rho_hat) + (1 - sparsity_target) * tf.log((1 - sparsity_target) / (1 - rho_hat)))

                            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda

                            loss = tf.reduce_mean(tf.square(X - decoder)) + sparsity_weight * sparsity_loss + l2_loss

                            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                            grads_and_vars = optimizer.compute_gradients(loss)
                            train_op = optimizer.apply_gradients(grads_and_vars)

                            init = tf.global_variables_initializer()

                            with tf.Session() as sess:
                                sess.run(init)
                                total_batch = int(len(allDataN) / batch_size)
                                for epoch in range(training_epochs):
                                    avg_cost = 0
                                    for i in range(total_batch):
                                        batch_x = allDataN[i * batch_size: (i + 1) * batch_size]
                                        _, c, gradients = sess.run([train_op, loss, grads_and_vars], feed_dict={X: batch_x})
                                        avg_cost += c / total_batch
                                    gradient_norm = np.sqrt(np.sum([np.sum(grad**2) for grad, _ in gradients]))
                                    if gradient_norm >= .55 and epoch > 525:
                                        #print("Minimum gradient threshold reached. Stopping training. Epoch:", '%04d' % (epoch))
                                        break
                                    #if epoch % display_step == 0:
                                        #print("Gradient: ", '{:.9f}'.format(gradient_norm))
                                        #print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(avg_cost))
                                        # Check if loss falls below threshold
                                #print("Optimization Finished!")

                                encoded_data = sess.run(encoder, feed_dict={X: allDataN})

                            y = data.iloc[:, -1].values
                            #knn = KNeighborsClassifier(n_neighbors=11)
                            #knn.fit(encoded_data, y)

                            #predictions = knn.predict(encoded_data)

                            #accuracy = np.mean(predictions == y)
                            #print("Accuracy:", accuracy)

                            if np.isnan(encoded_data).all():
                                encoded_data = np.zeros_like(encoded_data)

                            elif np.isnan(encoded_data).any():
                                encoded_data = np.nan_to_num(encoded_data, nan=np.nanmean(encoded_data))

                            if np.isnan(y).any():
                                y = np.nan_to_num(y, nan=np.nanmean(y))

                            db_index = davies_bouldin_index(encoded_data, y)
                            accuracies.append(db_index)

                            #accuracies.append(accuracy)

                            #encoded_df = pd.DataFrame(encoded_data, columns=[f'feature_{i}' for i in range(encoded_data.shape[1])])
                            #3encoded_df['outcome'] = data.iloc[:, -1].values

                            #mean_encoded_values = encoded_df.groupby('outcome').mean()

                            #mean_diff = mean_encoded_values.diff(axis=0).abs().sum().sum()  # Sum of absolute differences

                            #max_mean_diff = mean_encoded_values.max(axis=0) - mean_encoded_values.min(axis=0)
                            #overlap_metric = 1 - mean_diff / max_mean_diff

                            #print("Overlap Metric:", overlap_metric)
                            #accuracies.append(overlap_metric)

                        avg_accuracy = np.mean(accuracies)

                        results[(batch_size, learning_rate, n_neighbors, sparsity_target, sparsity_weight, l2_lambda)] = avg_accuracy

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

top_10_combinations = sorted_results[:10]
print("Top 10 combinations:")
for i, (params, accuracy) in enumerate(top_10_combinations):
    print(f"Combination {i+1}: Batch Size = {params[0]}, Learning Rate = {params[1]}, n_neighbors = {params[2]}, sparsity_target = {params[3]}, sparsity_weight = {params[4]}, l2_lambda = {params[5]}, Accuracy = {accuracy}")