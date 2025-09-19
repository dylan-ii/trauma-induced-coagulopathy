import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

data = pd.read_excel("thrombosisIPA.xlsx")

X = data.iloc[:, 8:40]
y = data.iloc[:, -1]

def custom_to_numeric(x):
    if isinstance(x, str) and re.match(r'^-?\d+(\.\d+)?\*$', x):
        return float(re.sub(r'\*$', '', x))
    return pd.to_numeric(x, errors='coerce')

X.fillna(data.rolling(window=3, min_periods=1, axis=0).mean(), inplace=True)

X = X.applymap(custom_to_numeric)

for col in X.columns:
    if X[col].var() == 0:
        X.drop(columns=col, inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=37)

#parameters for grid search
param_grid = {
    'n_estimators': [200],#[100, 200, 300],
    'max_depth': [3],#[3, 5, 7],
    'learning_rate': [0.05],#[0.01, 0.05, 0.1],
    'subsample': [0.9],#[0.5, 0.7, 0.9],
    'colsample_bytree': [0.7]#[0.5, 0.7, 0.9],
}

xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=37)

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Accuracy:", best_score)

#best_model = grid_search.best_estimator_
#y_pred = best_model.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print("Test Accuracy:", accuracy)
