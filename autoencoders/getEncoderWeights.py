import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import re

tf.disable_v2_behavior()

filename = 'thrombosisIPA.xlsx'
data = pd.read_excel(filename)

def custom_to_numeric(x):
    if isinstance(x, str) and re.match(r'^-?\d+(\.\d+)?\*$', x):
        return float(re.sub(r'\*$', '', x))
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

# autoencoder parameters
hidden_size = 2
learning_rate = 0.3
training_epochs = 550
batch_size = 8
display_step = 100
l2_lambda = 0.007
sparsity_target = 0.5
sparsity_weight = 5
n_neighbors = 3

num_trials = 10

encoder_weights_list = []
accuracies = []

for trial in range(num_trials):
    print(f"Running trial {trial + 1}/{num_trials}")
    
    tf.reset_default_graph()
    tf.set_random_seed(trial + 1)
    
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

    encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder']), biases['encoder']))

    decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, weights['decoder']), biases['decoder']))

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

        encoded_data = sess.run(encoder, feed_dict={X: allDataN})
        
        encoder_weights = sess.run(weights['encoder'])
        
        encoder_weights_list.append(encoder_weights)
        
        X_train, X_test, y_train, y_test = train_test_split(encoded_data, data.iloc[:, -1].values, test_size=0.4, random_state=trial+1)
        
        y = data.iloc[:, -1].values
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(encoded_data, y)
        
        predictions = knn.predict(X_test)
        
        accuracy = np.mean(predictions == y_test)
        #accuracies.append(accuracy)
        print(f"Trial {trial + 1} Accuracy: {accuracy}")

encoder_weights_array = np.array(encoder_weights_list)

average_encoder_weights = np.mean(encoder_weights_array, axis=0)

average_abs_weights = np.mean(np.abs(average_encoder_weights), axis=1)

sorted_indices = np.argsort(average_abs_weights)[::-1]
sorted_weights = average_abs_weights[sorted_indices]
sorted_features = data.columns[8:40][sorted_indices]

top_features = sorted_features[:32]
top_weights = sorted_weights[:32]

plt.figure(figsize=(12, 8))
bars = plt.bar(range(len(top_features)), top_weights, align='center', alpha=0.8, color='red')

for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, top_features[i],
             ha='center', va='bottom', rotation=90, fontsize=18)

plt.title('Distribution of Absolute Weight of Input Features', fontsize=22)
plt.xlabel('Input Features', fontsize=18, labelpad=20)
plt.ylabel('Average Absolute Weight', fontsize=18, labelpad=20)
plt.ylim(0, max(top_weights) * 1.2)
#plt.tight_layout()
plt.show()

print(f"Average Accuracy of kNN Classifier on Encoded Data: {np.mean(accuracies)}")
