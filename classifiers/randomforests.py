import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

data = pd.read_excel("thrombosisIPA.xlsx")

#10,20,38
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=102)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), slice(0, 32))
    ])

rf_classifier = RandomForestClassifier()

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', rf_classifier)])

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

y_pred = grid_search.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
