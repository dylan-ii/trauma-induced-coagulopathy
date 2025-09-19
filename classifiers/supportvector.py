import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

data = pd.read_excel('thrombosisIPA.xlsx')
X = data.iloc[:, 8:40] 
y = data.iloc[:, -1]

def custom_to_numeric(x):
    if isinstance(x, str) and re.match(r'^-?\d+(\.\d+)?\*$', x):
        return float(re.sub(r'\*$', '', x))
    return pd.to_numeric(x, errors='coerce')

data.fillna(data.rolling(window=6, min_periods=1, axis=0).mean(), inplace=True)

data = data.applymap(custom_to_numeric)

for col in data.columns:
    if data[col].var() == 0:
        data.drop(columns=col, inplace=True)

minA = data.iloc[:, 8:40].min()
maxA = data.iloc[:, 8:40].max()
X_scaled = (data.iloc[:, 8:40] - minA) / (maxA - minA)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=39)

smote = SMOTE(random_state=39)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

from sklearn.model_selection import GridSearchCV

# parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],
    'class_weight': [None, 'balanced'],
    'tol': [1e-3, 1e-4, 1e-5],
    'max_iter': [-1, 1000, 2000],  # -1 for no limit

}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)