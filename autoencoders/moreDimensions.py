import tensorflow as tf
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

def custom_to_numeric(x):
    if isinstance(x, str) and re.match(r'^-?\d+(\.\d+)?\*$', x):
        return float(re.sub(r'\*$', '', x))
    return pd.to_numeric(x, errors='coerce')

# Load data from the Excel file
xls = pd.ExcelFile('AllData.xlsx')

# Load "Blood Biomarkers" sheet (assuming it's the first sheet)
data_biomarkers = pd.read_excel(xls, sheet_name='Blood Biomarkers')

# Load "Proteomics" sheet (assuming it's the second sheet)
data_proteomics = pd.read_excel(xls, sheet_name='Proteomics')

# Extract labels from "Blood Biomarkers" sheet (assuming last row is labels)
data_biomarkers_transposed = data_biomarkers.transpose()
labels = data_biomarkers_transposed.iloc[1:62, -1].values

# Convert labels to a suitable format for classification
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Transpose the "Proteomics" data to have objects as rows and features as columns
data_proteomics_transposed = data_proteomics.transpose()

# Select the desired ranges of data
proteomicsFrame = data_proteomics_transposed.iloc[1:62, 1:101] # Include all

# Apply custom conversion function to handle asterisks and numeric conversion
columns_of_interest = proteomicsFrame.applymap(custom_to_numeric)

# Drop columns with zero variance
columns_to_drop = []
for col in columns_of_interest.columns:
    if columns_of_interest[col].var() < 0.001:
        columns_to_drop.append(col)
columns_of_interest.drop(columns=columns_to_drop, inplace=True)

# Preprocess the data
minA = columns_of_interest.min()
maxA = columns_of_interest.max()
data_normalized = (columns_of_interest - minA) / (maxA - minA)
data_normalized.fillna(data_normalized.rolling(window=6, min_periods=1, axis=0).mean(), inplace=True)

# Convert DataFrame to numpy array
data_array = data_normalized.values

# Parameters    
input_size = data_array.shape[1]
hidden_size = 10  # Updated hidden size

# Define the encoder model with He normal initialization and dropout regularization
encoder_inputs = tf.keras.layers.Input(shape=(input_size,))
encoder_regularized = tf.keras.layers.Dense(hidden_size, activation='tanh',
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(encoder_inputs)
encoder_dropout = tf.keras.layers.Dropout(0.2)(encoder_regularized)
encoder_model = tf.keras.models.Model(inputs=encoder_inputs, outputs=encoder_dropout)

# Define the decoder model with He normal initialization and dropout regularization
decoder_inputs = tf.keras.layers.Input(shape=(hidden_size,))
decoder_regularized = tf.keras.layers.Dense(input_size, activation='tanh',
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(decoder_inputs)
decoder_dropout = tf.keras.layers.Dropout(0.2)(decoder_regularized)
decoder_model = tf.keras.models.Model(inputs=decoder_inputs, outputs=decoder_dropout)

# Combine encoder and decoder into an autoencoder model
autoencoder_outputs = decoder_model(encoder_model(encoder_inputs))
autoencoder = tf.keras.models.Model(inputs=encoder_inputs, outputs=autoencoder_outputs)

# Compile the model
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')

# Train the model                                     
num_epochs = 400
history = autoencoder.fit(data_array, data_array, epochs=num_epochs, verbose=1)

# Get encoded data
encoded_data = encoder_model.predict(data_array)

# Normalization
minA = encoded_data.min()
maxA = encoded_data.max()
encoded_scaled = (encoded_data[:] - minA) / (maxA - minA)

# Split data for training and evaluation
X_train, X_test, y_train, y_test = train_test_split(encoded_scaled, labels, test_size=0.3, random_state=1)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'], # Linear is much nicer
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],
    'class_weight': [None, 'balanced'],
    'tol': [1e-3, 1e-4, 1e-5],
    'max_iter': [-1, 1000, 2000],  # -1 for no limit
}

# Create GridSearchCV object
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit the Grid Search
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Predict on the test set with the best estimator
y_pred = grid_search.best_estimator_.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred, zero_division=0))

# Plot encoded data
colors = ['red' if label == 1 else 'black' for label in labels]
plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=colors, s=26)
plt.xlabel('Encoded Dimension 1', fontsize=18, labelpad=20)
plt.ylabel('Encoded Dimension 2', fontsize=18, labelpad=20)
plt.title('Encoded Data as a Scatter Plot with Labels', fontsize=22)

# Create a legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='1', markerfacecolor='red', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='0', markerfacecolor='black', markersize=10)
]
plt.legend(handles=legend_elements, title='Mortality', loc='best')

plt.show()