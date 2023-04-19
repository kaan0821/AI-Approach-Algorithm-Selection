import argparse
import joblib
from model import Net
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as f
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from reg_hypertune import find_best_reg_hyperparams
from class_hypertune import find_best_class_hyperparams
from forest_hypertune import find_best_forest_hyperparams

def main():
    parser = argparse.ArgumentParser(description="Train an AS model and save it to file")
    parser.add_argument("--model-type", type=str, required=True, help="Path to a trained AS model (a .pt file)")
    parser.add_argument("--data", type=str, required=True, help="Path to a dataset")
    parser.add_argument("--save", type=str, required=True, help="Save the trained model (and any related info) to a .pt file")
    
    args = parser.parse_args()

    print(f"\nTraining a {args.model_type} model on {args.data}, and save it to {args.save}")
    
    # Loading data from the txt files
    instance_data = np.loadtxt(f"{args.data}instance-features.txt")
    performance_data = np.loadtxt(f"{args.data}performance-data.txt")
    # Normalizing instance data
    mean = np.mean(instance_data, axis=0)
    std = np.std(instance_data, axis=0)
    epsilon = 1e-8  # Small constant to avoid division by zero error
    normalized_instance_data = (instance_data - mean) / (std + epsilon)

    # Split the data into training set and validation set with an 80-20 split
    num_instances = normalized_instance_data.shape[0]
    split_idx = int(num_instances * 0.8)  # index to split the data
    indices = np.random.permutation(num_instances)  # randomize the indices
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    x_train, x_val = normalized_instance_data[train_indices], normalized_instance_data[val_indices]
    y_train, y_val = performance_data[train_indices], performance_data[val_indices]


    # ------------ Part 1 Regression NN ---------------
    if args.model_type == "regresion_nn":
        # Get the best hyperparameters using the find_best_hyperparams function
        #best_params = find_best_reg_hyperparams(x_train, y_train)
        #print(f"Best parameters found: {best_params}")

        # Set up hyperparameters
        input_size = len(x_train[0])
        output_size = len(y_train[0])
        learning_rate = 0.01
        hidden_size = 64
        max_epochs = 100
        batch_size = 50

        # Initialize the model and optimizer
        model = Net(input_size, output_size, hidden_size)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Define the loss function
        loss_function = nn.MSELoss()
        # Initialize lists to store accuracy and validation accuracy values
        train_accuracies = []
        val_accuracies = []
        val_losses = []
        best_val_loss = np.inf
        epochs_since_improvement = 0
        epoch_count = 0
        # Train the model
        for epoch in range(max_epochs):
            total_corrects = 0
            # Set the model to training mode
            model.train()
            epoch_count += 1
            # Loop over the training data in batches
            for i in range(0, len(x_train), batch_size):
                # Get a batch of training data
                x = torch.tensor(x_train[i:i+batch_size], dtype=torch.float32)
                y_true = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32)
                
                # Forward pass
                y_pred = model(x)

                # Calculate the number of correct predictions and update the total
                predicted_classes = y_pred.argmin(axis=1)
                true_classes = y_true.argmin(axis=1)
                n_corrects = (predicted_classes == true_classes).sum()

                # Zero the gradients
                optimizer.zero_grad()
                
                # Compute the loss
                loss = loss_function(y_pred, y_true)
                # Backward pass and optimization
                loss.backward()
                # Update the current model using the calculated gradients
                optimizer.step()
                # Update statistics
                total_corrects += n_corrects

            # Calculate the accuracy
            accuracy = total_corrects / len(train_indices)
            # Evaluate the model on the validation set
            model.eval()
            with torch.no_grad():
                x_val = torch.tensor(x_val, dtype=torch.float32)
                y_val_true = torch.tensor(y_val, dtype=torch.float32)
                y_val_pred = model(x_val)
                val_loss = loss_function(y_val_pred, y_val_true)
                # Calculate validation accuracy
                predicted_val_classes = y_val_pred.argmin(axis=1)
                true_val_classes = y_val_true.argmin(axis=1)
                val_corrects = (predicted_val_classes == true_val_classes).sum()
                val_accuracy = val_corrects / len(val_indices)
            
            # Print the loss and validation loss for this epoch
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Accuracy: {accuracy: 3.4f}")
            # Append accuracy and validation accuracy to the lists
            train_accuracies.append(accuracy)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss.item())
            # Early Stopping Mechanism
            # Check if the validation loss has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                
            # Check if training should be stopped due to lack of improvement in validation loss
            if epochs_since_improvement >= 10:
                print(f"No improvement in validation loss for 10 epochs, early stopping training...")
                break

        # print results
        print(f"\nTraining finished")
        # Save the trained model and hidden layer size to a file
        metadata = {
            "state_dict": model.state_dict(),
            "hidden_size": hidden_size,
            "model_type" : args.model_type
        }
        torch.save(metadata, f"{args.save}")
        print(f"\nTrained model saved to {args.save}")
        plotAccuracy(epoch_count,train_accuracies,val_accuracies)
        plotLoss(epoch_count,val_losses)

    # --------------- Part 2 & 3 Classification NN ---------------
    elif args.model_type == "classification_nn" or args.model_type == "classification_nn_cost":
        #best_params = find_best_class_hyperparams(x_train, y_train)
        #print(f"Best parameters found: {best_params}")
        # Set up hyperparameters
        input_size = len(x_train[0])
        output_size = len(y_train[0])
        learning_rate = 0.001
        hidden_size = 64
        max_epochs = 100
        batch_size = 128

        # Initialize the model and optimizer
        model = Net(input_size, output_size, hidden_size)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
        # Define the loss function
        loss_function = nn.CrossEntropyLoss()

        # Initialize lists to store accuracy and validation accuracy values
        train_accuracies = []
        val_accuracies = []
        val_losses = []
        best_val_loss = np.inf
        epochs_since_improvement = 0
        epoch_count = 0
        regret_control = 1
        # Train the model
        for epoch in range(max_epochs):
            total_corrects = 0
            epoch_count += 1
            # Set the model to training mode
            model.train()
            
            # Loop over the training data in batches
            for i in range(0, len(x_train), batch_size):
                # Get a batch of training data
                x = torch.tensor(x_train[i:i+batch_size], dtype=torch.float32)
                y_true = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32)
                
                # Forward pass
                y_pred = model(x)

                # Calculate the number of correct predictions and update the total
                predicted_classes = f.softmax(y_pred,dim=1).argmax(axis=1)
                true_classes = y_true.argmin(axis=1)
                n_corrects = (predicted_classes == true_classes).sum()

                # Calculate regret if needed
                if args.model_type == "classification_nn_cost":
                    predicted_costs = y_true[torch.arange(y_true.shape[0]), predicted_classes]
                    true_best_costs = y_true[torch.arange(y_true.shape[0]), true_classes]
                    regret = (predicted_costs - true_best_costs).mean()

                # Zero the gradients
                optimizer.zero_grad()
                
                # Compute the loss
                loss = loss_function(y_pred, true_classes)
                if args.model_type == "classification_nn_cost":
                    # Add the regret term to the loss
                    loss_with_regret = loss + regret
                    # Backward pass and optimization
                    loss_with_regret.backward()
                else:
                    # Backward pass and optimization
                    loss.backward()
                # Update the current model using the calculated gradients
                optimizer.step()
                # Update statistics
                total_corrects += n_corrects

            # Calculate the accuracy
            accuracy = total_corrects / len(train_indices)
            # Evaluate the model on the validation set
            model.eval()
            with torch.no_grad():
                x_val = torch.tensor(x_val, dtype=torch.float32)
                y_val_true = torch.tensor(y_val, dtype=torch.float32)
                y_val_pred = model(x_val)
                val_loss = loss_function(y_val_pred, y_val_true.argmin(axis=1))
                # Calculate validation accuracy
                predicted_val_classes = f.softmax(y_val_pred,dim=1).argmax(axis=1)
                true_val_classes = y_val_true.argmin(axis=1)
                # Calculate regret if needed
                if args.model_type == "classification_nn_cost":
                    val_predicted_costs = y_val_true[torch.arange(y_val_true.shape[0]), predicted_val_classes]
                    val_true_best_costs = y_val_true[torch.arange(y_val_true.shape[0]), true_val_classes]
                    regret = (val_predicted_costs - val_true_best_costs).mean()
                    val_loss = val_loss + regret_control * regret
                val_corrects = (predicted_val_classes == true_val_classes).sum()
                val_accuracy = val_corrects / len(val_indices)
            
            # Print the loss and validation loss for this epoch
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Accuracy: {accuracy: 3.4f}")
            # Append accuracy and validation accuracy to the lists
            train_accuracies.append(accuracy)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss.item())
            # Early Stopping Mechanism
            # Check if the validation loss has improved
            if val_loss < best_val_loss - 0.001:
                best_val_loss = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                
            # Check if training should be stopped due to lack of improvement in validation loss
            if epochs_since_improvement >= 10:
                print(f"No improvement in validation loss for 10 epochs, early stopping training...")
                break

        # print results
        print(f"\nTraining finished")
        # Save the trained model and hidden layer size to a file
        metadata = {
            "state_dict": model.state_dict(),
            "hidden_size": hidden_size,
            "model_type" : args.model_type
        }
        torch.save(metadata, f"{args.save}")
        print(f"\nTrained model saved to {args.save}")
        plotAccuracy(epoch_count,train_accuracies,val_accuracies)
        plotLoss(epoch_count,val_losses)

    # ------------- Part 3 Pairwise Classification NN -------------
    elif args.model_type == "pairwise_classifier":
        # Train binary classifiers for each pair of algorithms
        num_algorithms = y_train.shape[1]
        pairwise_classifiers = []
        # Get all combinations of pair algos
        for i, j in itertools.combinations(range(num_algorithms), 2):
            classifier = pairwise_classifier(x_train, y_train, i, j)
            pairwise_classifiers.append((i, j, classifier))

        # Save the state dicts of all pairwise classifiers after training
        saved_state_dicts = []
        for idx1, idx2, classifier in pairwise_classifiers:
            saved_state_dicts.append({
                'indices': (idx1, idx2),
                'state_dict': classifier.state_dict()
            })

        # print results
        print(f"\nTraining finished")
        # Save the trained model and hidden layer size to a file
        metadata = {
            "state_dict": saved_state_dicts,
            "hidden_size": 64,
            "model_type" : args.model_type
        }
        torch.save(metadata, f"{args.save}")
        print(f"\nTrained model saved to {args.save}")

    # ------------- Part 4 Random Forest Model -------------
    elif args.model_type == "random_forest":

        best_algorithms = np.argmin(y_train, axis=1)
        #best = find_best_forest_hyperparams(x_train,best_algorithms)
        # Train a Random Forest Classifier & Regressor
        model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2,random_state=42)
        model2 = RandomForestRegressor(n_estimators=100,random_state=42)
        model.fit(x_train, best_algorithms)
        model2.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        y_train_pred2 = model2.predict(x_train)
        # Calculate the accuracy of the model on the training set
        train_accuracy = accuracy_score(best_algorithms, y_train_pred)
        print(f"Random Forest Classifier Accuracy on the training set: {train_accuracy:.4f}")
        # Predict the best algorithm on the validation set
        y_val_pred = model.predict(x_val)
        y_val_pred2 = model2.predict(x_val)
        best_algorithms_val = np.argmin(y_val, axis=1)
        # Calculate the accuracy of the model
        accuracy = accuracy_score(best_algorithms_val, y_val_pred)
        print(f"Random Forest Classifier Accuracy on the validation set: {accuracy:.4f}")
        # Accuracy for Regression
        train_corrects = (y_train_pred2.argmin(axis=1) == best_algorithms).sum()
        train_accuracy2 = train_corrects / len(train_indices)
        val_corrects = (y_val_pred2.argmin(axis=1) == best_algorithms_val).sum()
        accuracy2 = val_corrects / len(val_indices)
        print("--------------------------------------------------------")
        print(f"Random Forest Regression Accuracy on the training set: {train_accuracy2:.4f}")
        print(f"Random Forest Regression Accuracy on the validation set: {accuracy2:.4f}")
        metadata = {
            "model_type" : args.model_type
        }
        torch.save(metadata, f"{args.save}")
        joblib.dump(model, "models/part4.pkl")
        joblib.dump(model2, "models/part4_2.pkl")
        print(f"\nTrained model saved to {args.save}")
    else:
        print("Illegal argument")



def plotAccuracy(epochs,train_acc,val_acc):
    # Plot the training and validation accuracy
    plt.plot(range(1, epochs + 1), train_acc, label='Training Accuracy', color='blue')
    plt.plot(range(1, epochs + 1), val_acc, label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plotLoss(epochs,val_loss):
    # Plot the training and validation accuracy
    plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.show()

def pairwise_classifier(x_train, y_train, idx1, idx2, regret_control = 1, max_epochs=100, hidden_size=64, learning_rate=0.001, batch_size=128):
    # Create binary labels for this pair of algorithms
    print(f"Training classifier for algo:{idx1} and algo:{idx2}...")
    # Gets an array of which one is bigger or smaller
    binary_labels = (y_train[:, idx1] < y_train[:, idx2]).astype(np.int64)

    # Initialize the model and optimizer
    input_size = x_train.shape[1]
    model = Net(input_size, 2, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    loss_function = nn.CrossEntropyLoss()
    # Calculate the regrets for the entire dataset
    regrets = torch.tensor(y_train[:, idx1] - y_train[:, idx2], dtype=torch.float32)
    # Train the model
    for epoch in range(max_epochs):
        for i in range(0, len(x_train), batch_size):
            end_idx = min(i+batch_size, len(x_train))
            x = torch.tensor(x_train[i:end_idx], dtype=torch.float32)
            y_true = torch.tensor(binary_labels[i:end_idx], dtype=torch.int64)
            y_costs = torch.tensor(y_train[i:end_idx], dtype=torch.float32)
            
            # Forward pass
            y_pred= model(x)

            # Calculate the loss
            loss = loss_function(y_pred, y_true)

            # Calculate the regret
            # Get the indices of the predicted algorithm and true best algorithm
            predicted_algo_idx = torch.where(y_true == 1, idx1, idx2)
            true_best_algo_idx = y_costs.argmin(dim=1)
            # Calculate the regret term
            regret = torch.zeros_like(y_true, dtype=torch.float32)
            # Boolean tensor containing wrongly predicted algo indices
            mask = (predicted_algo_idx != true_best_algo_idx)
            # Calculate the regret only for the instances with a mismatch between predicted and true best algorithms
            regret[mask] = y_costs[mask, predicted_algo_idx[mask]] - y_costs[mask, true_best_algo_idx[mask]]
            regret_loss = torch.mean(regret)

            # Compute the total loss as a combination of the original loss and the regret term
            total_loss = loss + regret_control * regret_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return model

if __name__ == "__main__":
    main()
