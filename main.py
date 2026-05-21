"""
Main execution script for the GMM Spoken Digit Recognition pipeline.
Trains models on the training dataset and evaluates them on the evaluation dataset.
"""

import os
from functions import group_files_by_digit, train_final_models, generate_predictions_csv
from evaluate import evaluate

def main():
    # Define dataset and output paths
    train_folder = "digits/train"
    eval_folder = "digits/eval"
    results_file = "results.csv"

    # Quick validation to prevent confusing errors if folders are missing
    if not os.path.exists(train_folder) or not os.path.exists(eval_folder):
        print(f"Error: Dataset folders not found. Ensure '{train_folder}' and '{eval_folder}' exist.")
        return

    # Optimal hyperparameters found via grid search
    best_params = {
        "n_mfcc": 8,
        "delta_mode": 1,
        "norm_mode": 1,
        "n_components": 2,
        "covariance_type": "full",
        "n_mels": 8
    }

    print("--- Starting GMM Spoken Digit Recognition Pipeline ---")
    
    print(f"\n1. Grouping training files from '{train_folder}'...")
    grouped_train_data = group_files_by_digit(train_folder)

    print("\n2. Training final GMM models...")
    models = train_final_models(
        folder_path=train_folder,
        grouped_data=grouped_train_data,
        n_mfcc=best_params["n_mfcc"],
        n_mels=best_params["n_mels"],
        delta_mode=best_params["delta_mode"],
        norm_mode=best_params["norm_mode"],
        n_components=best_params["n_components"],
        covariance_type=best_params["covariance_type"]
    )

    print(f"\n3. Generating predictions for evaluation data in '{eval_folder}'...")
    generate_predictions_csv(
        folder_path=eval_folder,
        models=models,
        output_csv=results_file,
        n_mfcc=best_params["n_mfcc"],
        n_mels=best_params["n_mels"],
        delta_mode=best_params["delta_mode"],
        norm_mode=best_params["norm_mode"]
    )

    print("\n4. Evaluating predictions...")
    try:
        evaluate(results_file)
    except Exception as e:
        print(f"Evaluation failed. Make sure the true labels are correctly configured. Error: {e}")

    print("\n--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()