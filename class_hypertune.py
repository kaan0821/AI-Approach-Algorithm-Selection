import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import Net
from skorch import NeuralNetClassifier
from sklearn.model_selection import RandomizedSearchCV


def find_best_class_hyperparams(x_train, y_train):
    print("Finding best hyper-parameters...")
    input_size = x_train.shape[1]
    output_size = y_train.shape[1]

    # Find the best algorithm indices (class labels) for each instance
    y_train_best_algo = np.argmin(y_train, axis=1) 

    net = NeuralNetClassifier(
        Net,
        module__input_size = input_size,
        module__output_size = output_size,
        optimizer = optim.Adam,
        criterion = nn.CrossEntropyLoss,
        max_epochs = 100,
        train_split = None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose = 0
    )

    param_dist = {
        'lr': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'optimizer__weight_decay': [0, 1e-4, 1e-3, 1e-2],
        'module__hidden_size': [32, 64, 128, 256, 512],
        'batch_size': [16, 32, 64, 128],
    }

    random_search = RandomizedSearchCV(
        net,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        verbose=0,
        n_jobs=-1,
        error_score='raise'
    )

    random_search.fit(np.array(x_train, dtype=np.float32), y_train_best_algo.astype(np.int64))

    best_params = random_search.best_params_
    return best_params
