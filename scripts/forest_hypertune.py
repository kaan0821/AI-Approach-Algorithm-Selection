import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def find_best_forest_hyperparams(x_train,y_train):
    print("Finding best hyper-parameters...")
    # Define the parameter grid for the RandomForestClassifier
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Create a RandomForestClassifier instance
    rf_clf = RandomForestClassifier(random_state=42)

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(x_train, y_train)

    # Get the best parameters found by GridSearchCV
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    return best_params

