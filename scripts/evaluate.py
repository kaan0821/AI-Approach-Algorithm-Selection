import argparse
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as f
from model import Net
import numpy as np
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained AS model on a test set")
    parser.add_argument("--model", type=str, required=True, help="Path to a trained AS model (a .pt file)")
    parser.add_argument("--data", type=str, required=True, help="Path to a dataset")
    
    args = parser.parse_args()

    print(f"\nLoading trained model {args.model} and evaluating it on {args.data}")
    
    # Load the test data
    instance_data = np.loadtxt(f"{args.data}/instance-features.txt")
    performance_data = np.loadtxt(f"{args.data}/performance-data.txt")

    # Normalize the instance data
    mean = np.mean(instance_data, axis=0)
    std = np.std(instance_data, axis=0)
    epsilon = 1e-8  # Small constant to avoid division by zero error
    normalized_instance_data = (instance_data - mean) / (std + epsilon)
    metadata = torch.load(args.model)
    # ------- Evaluate Regression -------
    if metadata["model_type"] == "regresion_nn":       
        # Load the trained model and its hidden layer size
        hidden_size = metadata["hidden_size"]
        model = Net(normalized_instance_data.shape[1], performance_data.shape[1], hidden_size)
        model.load_state_dict(metadata["state_dict"])

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            x_test = torch.tensor(normalized_instance_data, dtype=torch.float32)
            y_test_true = torch.tensor(performance_data, dtype=torch.float32)
            y_test_pred = model(x_test)
            test_loss = nn.MSELoss()(y_test_pred, y_test_true)
            
        # Calculate average loss
        avg_loss = test_loss.item() # the average loss value across the given dataset

        # Make predictions and evaluate the model's performance
        y_pred = y_test_pred.numpy()
        y_true = y_test_true.numpy()
        pred_algorithm = y_pred.argmin(axis=1)
        true_algorithm = y_true.argmin(axis=1)
        accuracy = np.mean(pred_algorithm == true_algorithm) # classification accuracy
        predicted_costs = y_true[np.arange(len(true_algorithm)), pred_algorithm]
        avg_cost = np.mean(predicted_costs)
        # Calculate SBS cost
        avg_costs = np.mean(y_true, axis=0)
        sbs_idx = np.argmin(avg_costs)
        sbs_avg_cost = np.mean(y_true[:, sbs_idx])
        # Calculate VBS cost
        best_idx = np.argmin(y_true, axis=1)
        best_costs = y_true[np.arange(len(y_true)), best_idx]
        vbs_avg_cost = np.mean(best_costs)
        # Calculate SBS-VBS Gap
        sbs_vbs_gap = (avg_cost - vbs_avg_cost)/(sbs_avg_cost - vbs_avg_cost) # the SBS-VBS gap of your model on the given dataset
        # print results
        print(f"\nFinal results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")
    
     # ------- Evaluate Classification -------
    elif metadata["model_type"] == "classification_nn" or metadata["model_type"] == "classification_nn_cost":
        # Load the trained model and its hidden layer size
        hidden_size = metadata["hidden_size"]
        model = Net(normalized_instance_data.shape[1], performance_data.shape[1], hidden_size)
        model.load_state_dict(metadata["state_dict"])

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            x_test = torch.tensor(normalized_instance_data, dtype=torch.float32)
            y_test_true = torch.tensor(performance_data, dtype=torch.float32)
            y_test_pred = model(x_test)
            test_loss = nn.CrossEntropyLoss()(y_test_pred, y_test_true.argmin(axis=1))
            
        # Calculate average loss
        avg_loss = test_loss.item() # the average loss value across the given dataset

        # Make predictions and evaluate the model's performance
        y_pred = y_test_pred.numpy()
        y_pred_tensor = torch.tensor(y_pred)
        y_true = y_test_true.numpy()
        pred_algorithm = f.softmax(y_pred_tensor,dim=1).argmax(axis=1)
        true_algorithm = y_true.argmin(axis=1)
        accuracy = np.mean(pred_algorithm.numpy() == true_algorithm) # classification accuracy
        # Select the predicted costs for each instance using the predicted algorithm indices
        predicted_costs = y_true[np.arange(len(true_algorithm)), pred_algorithm]
        avg_cost = np.mean(predicted_costs)

        # Calculate SBS cost
        avg_costs = np.mean(y_true, axis=0)
        sbs_idx = np.argmin(avg_costs)
        sbs_avg_cost = np.mean(y_true[:, sbs_idx])
        # Calculate VBS cost
        best_idx = np.argmin(y_true, axis=1)
        best_costs = y_true[np.arange(len(y_true)), best_idx]
        vbs_avg_cost = np.mean(best_costs)
        # Calculate SBS-VBS Gap
        sbs_vbs_gap = (avg_cost - vbs_avg_cost)/(sbs_avg_cost - vbs_avg_cost) # the SBS-VBS gap of your model on the given dataset
        # print results
        print(f"\nFinal results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")
   
   # -------------- Part 3 Pairwise Classification --------------- 
    elif metadata["model_type"] == "pairwise_classifier":
        # Load the saved state dicts
        saved_state_dicts = metadata["state_dict"]
        hidden_size = metadata["hidden_size"]
        x_test = torch.tensor(normalized_instance_data, dtype=torch.float32)
        y_test_true = torch.tensor(performance_data, dtype=torch.float32)
        num_algorithms = y_test_true.shape[1]
        # Recreate the classifiers and load the state dicts
        pairwise_classifiers = []
        for saved_state in saved_state_dicts:
            idx1, idx2 = saved_state['indices']
            classifier = Net(x_test.shape[1], 2, hidden_size)
            classifier.load_state_dict(saved_state['state_dict'])
            pairwise_classifiers.append((idx1, idx2, classifier))

        # Perform evaluation using the loaded classifiers
        votes = np.zeros((len(x_test), num_algorithms), dtype=int)
        for idx1, idx2, classifier in pairwise_classifiers:
            with torch.no_grad():
                y_test_pred = classifier(x_test)
                predicted_class = torch.argmax(y_test_pred, axis=1).numpy()
                # Update the votes for each algorithm
                votes[predicted_class == 1, idx1] += 1
                votes[predicted_class == 0, idx2] += 1

        # Choose the best algorithm based on the max vote
        best_predicted_algorithm = np.argmax(votes, axis=1)

        y_true = y_test_true.numpy()
        # Calculate the accuracy and other performance metrics
        true_algorithm = y_true.argmin(axis=1)
        accuracy = np.mean(best_predicted_algorithm == true_algorithm) # classification accuracy
        # Select the predicted costs for each instance using the predicted algorithm indices
        predicted_costs = y_true[np.arange(len(true_algorithm)), best_predicted_algorithm]
        avg_cost = np.mean(predicted_costs)

        # Calculate SBS cost
        avg_costs = np.mean(y_true, axis=0)
        sbs_idx = np.argmin(avg_costs)
        sbs_avg_cost = np.mean(y_true[:, sbs_idx])
        # Calculate VBS cost
        best_idx = np.argmin(y_true, axis=1)
        best_costs = y_true[np.arange(len(y_true)), best_idx]
        vbs_avg_cost = np.mean(best_costs)
        # Calculate SBS-VBS Gap
        sbs_vbs_gap = (avg_cost - vbs_avg_cost)/(sbs_avg_cost - vbs_avg_cost) # the SBS-VBS gap of your model on the given dataset
         # print results
        print(f"\nFinal results: accuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")
    
    # -------------- Part 4 Random Forest Model --------------- 
    elif metadata["model_type"] == "random_forest":
        # Evaluate on the test set
        x_test = normalized_instance_data
        y_test = performance_data
        # Load the model
        model = joblib.load(f"models/part4.pkl")
        model2 = joblib.load(f"models/part4_2.pkl")
        # Calculate the accuracy of the model on the test set
        best_algorithms_test = np.argmin(y_test, axis=1)
        y_test_pred = model.predict(x_test)
        y_test_pred2 = model2.predict(x_test)
        test_accuracy = accuracy_score(best_algorithms_test, y_test_pred)
        test_corrects2 = (y_test_pred2.argmin(axis=1) == best_algorithms_test).sum()
        test_accuracy2 = test_corrects2 / len(y_test)
        # Make predictions and evaluate the model's performance
        # Select the predicted costs for each instance using the predicted algorithm indices
        predicted_costs = y_test[np.arange(len(y_test)), y_test_pred]
        predicted_costs2 = y_test[np.arange(len(y_test)), y_test_pred2.argmin(axis=1)]
        avg_cost = np.mean(predicted_costs)
        avg_cost2 = np.mean(predicted_costs2)
        # Calculate SBS cost
        avg_costs = np.mean(y_test, axis=0)
        sbs_idx = np.argmin(avg_costs)
        sbs_avg_cost = np.mean(y_test[:, sbs_idx])
        # Calculate VBS cost
        best_idx = np.argmin(y_test, axis=1)
        best_costs = y_test[np.arange(len(y_test)), best_idx]
        vbs_avg_cost = np.mean(best_costs)
        # Calculate SBS-VBS Gap
        sbs_vbs_gap = (avg_cost - vbs_avg_cost)/(sbs_avg_cost - vbs_avg_cost) # the SBS-VBS gap of your model on the given dataset
        sbs_vbs_gap2 = (avg_cost2 - vbs_avg_cost)/(sbs_avg_cost - vbs_avg_cost)

        print(f"\nClassifier Final results: accuracy: {test_accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")
        print(f"\nRegressor Final results: accuracy: {test_accuracy2:4.4f}, \tavg_cost: {avg_cost2:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap2:2.4f}")
    else:
        print("Illegal argument")



if __name__ == "__main__":
    main()
