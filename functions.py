import os
import csv
import itertools
import numpy as np
import librosa
from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple, Optional, Any

def cms(X: np.ndarray) -> np.ndarray:
    """Cepstral Mean Subtraction."""
    means = np.mean(X, axis=1)
    return (X.T - means.T).T


def cmvn(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Cepstral Mean and Variance Normalization."""
    means = np.mean(X, axis=1)
    stdevs = np.std(X, axis=1)
    stdevs = np.where(stdevs < eps, eps, stdevs)
    return ((X.T - means) / stdevs).T




def extract_mfcc(
        y: np.ndarray, 
        fs: int,
        n_mfcc: int, 
        n_fft: int, 
        win_length: int, 
        hop_length: int, 
        n_mels: int, 
        first_feature_id: int,
        norm_mode: int = 1,
        delta_mode: int = 1
    ) -> np.ndarray:
    """Extracts MFCC features with optional normalization and delta coefficients."""
    
    mfcc = librosa.feature.mfcc(
        y=y, sr=fs,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels
    ).T

    mfcc = mfcc[:, first_feature_id:]

    # Normalization
    if norm_mode == 2:
        mfcc = cms(mfcc)
    elif norm_mode == 3:
        mfcc = cmvn(mfcc)

    # Delta and Delta-Delta coefficients
    if delta_mode == 2:
        delta = librosa.feature.delta(mfcc)
        mfcc = np.hstack([mfcc, delta])
    elif delta_mode == 3:
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc = np.hstack([mfcc, delta, delta2])

    return mfcc



def group_files_by_digit(folder_path: str) -> Dict[int, List[str]]:
    """Groups audio files by digit based on their filename structure."""
    data_files = os.listdir(folder_path)
    grouped_data = {
        digit: [f for f in data_files if f.endswith('.wav') and int(f.split('_')[1]) == digit]
        for digit in range(10)
    }
    return grouped_data

def create_mfcc_dict(
        folder_path: str, 
        grouped_data: Dict[int, List[str]],
        n_mfcc: int, 
        n_fft: int, 
        win_length: int, 
        hop_length: int, 
        n_mels: int, 
        first_feature_id: int,
        norm_mode: int, 
        delta_mode: int
    ) -> Dict[int, List[Tuple[str, np.ndarray]]]:
    """Creates a dictionary containing extracted MFCCs for each digit."""
    
    mfcc_dict = {}

    for digit, files in grouped_data.items():
        mfcc_list = []
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            audio_data, fs = librosa.load(file_path, sr=None)
            
            features = extract_mfcc(
                y=audio_data, fs=fs,
                n_mfcc=n_mfcc, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                n_mels=n_mels, first_feature_id=first_feature_id,
                norm_mode=norm_mode, delta_mode=delta_mode
            )
            mfcc_list.append((file_name, features))

        mfcc_dict[digit] = mfcc_list

    return mfcc_dict



def cross_validate_gmm(
        mfcc_dict: Dict[int, List[Tuple[str, np.ndarray]]],
        n_components: int = 8,
        covariance_type: str = 'full',
        n_folds: int = 5,
        random_state: int = 0
    ) -> Tuple[float, List[int], List[int]]:
    """Evaluates GMM performance using speaker-aware Stratified K-Fold cross-validation."""
    
    features_list = []
    labels_list = []
    speakers = [] 

    for digit, recordings in mfcc_dict.items():
        for file_name, features in recordings:
            features_list.append(features)
            labels_list.append(digit)
            # First two characters of the filename represent the speaker ID
            speakers.append(file_name[:2])   

    X = np.array(features_list, dtype=object)
    y = np.array(labels_list)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    all_true_labels = []
    all_predicted_labels = []

    for train_idx, test_idx in skf.split(X, speakers):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        training_dict = {}
        for label in np.unique(y_train):
            idx = np.where(y_train == label)[0]
            training_dict[label] = X_train[idx].tolist()

        fold_models = {}
        for digit, features in training_dict.items():
            X_concat = np.concatenate(features, axis=0)
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=random_state
            )
            gmm.fit(X_concat)
            fold_models[digit] = gmm

        for sample_features in X_test:
            best_score = -float('inf')
            predicted_digit = None

            for digit, model in fold_models.items():
                try:
                    score = model.score(sample_features)
                except ValueError:
                    # Fallback if covariance matrix calculation fails
                    score = -float('inf')

                if score > best_score:
                    best_score = score
                    predicted_digit = digit

            all_predicted_labels.append(predicted_digit)
        all_true_labels.extend(y_test)

    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    print(f"Cross-validation Accuracy: {accuracy:.4f}")

    return accuracy, all_true_labels, all_predicted_labels


def run_experiment(
        folder_path: str,
        n_mfcc: int = 12,
        n_fft: int = 512,
        win_length: int = 320,
        hop_length: int = 160,
        n_mels: int = 40,
        first_feature_id: int = 1,
        norm_mode: int = 1,
        delta_mode: int = 1,
        n_components: int = 8,
        covariance_type: str = 'diag',
        n_folds: int = 5,
        random_state: int = 42
    ) -> Tuple[float, List[int], List[int]]:
    """Runs a complete feature extraction and evaluation pipeline."""
    
    grouped_data = group_files_by_digit(folder_path)

    mfcc_dict = create_mfcc_dict(
        folder_path, grouped_data,
        n_mfcc, n_fft, win_length, hop_length, n_mels, first_feature_id,
        norm_mode, delta_mode
    )

    accuracy, y_true, y_pred = cross_validate_gmm(
        mfcc_dict,
        n_components=n_components,
        covariance_type=covariance_type,
        n_folds=n_folds,
        random_state=random_state
    )

    return accuracy, y_true, y_pred

def train_final_models(
        folder_path: str,
        grouped_data: Dict[int, List[str]],
        n_mfcc: int = 12,
        n_fft: int = 512,
        win_length: int = 320,
        hop_length: int = 160,
        n_mels: int = 40,
        first_feature_id: int = 1,
        norm_mode: int = 1,
        delta_mode: int = 1,
        n_components: int = 8,
        covariance_type: str = 'diag',
        random_state: int = 42
    ) -> Dict[int, GaussianMixture]:
    """Trains final GMM models for digits 0-9 using the entire dataset."""
    
    mfcc_dict = create_mfcc_dict(
        folder_path, grouped_data,
        n_mfcc, n_fft, win_length, hop_length, n_mels, first_feature_id,
        norm_mode, delta_mode
    )

    models = {}
    for digit in range(10):
        # Extract only the MFCC arrays from the tuples before concatenation
        X_concat = np.concatenate([features for _, features in mfcc_dict[digit]], axis=0)
        
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state
        )
        gmm.fit(X_concat)
        models[digit] = gmm
        print(f"Model for digit {digit} successfully trained.")

    print("All models have been trained!")
    return models

def predict_single_file(
        file_path: str, 
        models: Dict[int, GaussianMixture],
        n_mfcc: int = 12,
        n_fft: int = 512,
        win_length: int = 320,
        hop_length: int = 160,
        n_mels: int = 30,
        first_feature_id: int = 1,
        norm_mode: int = 1,
        delta_mode: int = 1
    ) -> Tuple[str, int, float]:
    """Predicts the spoken digit from a single audio file."""
    
    audio_data, fs = librosa.load(file_path, sr=None)
    filename = os.path.basename(file_path)
    
    features = extract_mfcc(
        y=audio_data, fs=fs,
        n_mfcc=n_mfcc, n_fft=n_fft, win_length=win_length, hop_length=hop_length, 
        n_mels=n_mels, first_feature_id=first_feature_id,
        norm_mode=norm_mode, delta_mode=delta_mode
    )

    best_score = -float('inf')
    predicted_digit = None

    for digit, gmm in models.items():
        try:
            current_score = gmm.score(features)
        except ValueError:
            current_score = -float('inf')

        if current_score > best_score:
            best_score = current_score
            predicted_digit = digit

    return filename, predicted_digit, round(float(best_score), 2)

def generate_predictions_csv(
        folder_path: str, 
        models: Dict[int, GaussianMixture],
        output_csv: str = "results.csv", 
        n_files: int = 200, 
        n_mfcc: int = 8, 
        delta_mode: int = 1, 
        norm_mode: int = 1, 
        n_mels: int = 8
    ) -> None:
    """Generates a CSV file containing predictions for a batch of audio files."""
    
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        for i in range(1, n_files + 1):
            file_name = f"{i:03}.wav"
            file_path = os.path.join(folder_path, file_name)

            if not os.path.exists(file_path):
                continue

            filename, digit, _ = predict_single_file(
                file_path=file_path,
                models=models,
                n_mfcc=n_mfcc,
                delta_mode=delta_mode,
                norm_mode=norm_mode,
                n_mels=n_mels
            )
            writer.writerow([filename, digit])

    print(f"Results successfully saved to '{output_csv}'.")

def perform_grid_search(
        folder_path: str, 
        csv_path: str = "hyperparameters_results.csv", 
        n_mfcc_list: List[int] = [12, 14, 16, 18, 20], 
        delta_mode_list: List[int] = [1, 2], 
        norm_mode_list: List[int] = [1, 2, 3], 
        n_components_list: List[int] = [2, 4, 8, 16, 32], 
        covariance_type_list: List[str] = ['full', 'diag'], 
        n_mels_list: List[int] = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    ) -> None:
    """Evaluates various hyperparameter combinations and saves results to a CSV."""
    
    all_combinations = list(itertools.product(
        n_mfcc_list, delta_mode_list, norm_mode_list, 
        n_components_list, covariance_type_list, n_mels_list
    ))

    total_iters = len(all_combinations)
    best_accuracy = -1
    best_params = None

    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "n_mfcc", "delta_mode", "norm_mode",
            "n_components", "covariance_type", "n_mels", "accuracy"
        ])

        for idx, combo in enumerate(all_combinations, start=1):
            n_mfcc, delta_mode, norm_mode, n_components, covariance_type, n_mels = combo
            print(f"[{idx}/{total_iters}] Testing: n_mfcc={n_mfcc}, delta={delta_mode}, "
                  f"norm={norm_mode}, GMM_comp={n_components}, "
                  f"cov_type={covariance_type}, mels={n_mels}")

            try:
                acc, _, _ = run_experiment(
                    folder_path=folder_path,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    delta_mode=delta_mode,
                    norm_mode=norm_mode,
                    n_components=n_components,
                    covariance_type=covariance_type
                )
            except Exception as e:
                print(f"Experiment failed: {e}")
                acc = -1.0 # Use -1.0 for failed runs instead of None to keep column numerical

            writer.writerow([n_mfcc, delta_mode, norm_mode, n_components, covariance_type, n_mels, acc])

            if acc > best_accuracy:
                best_accuracy = acc
                best_params = combo

    print("\nGrid search completed.")
    if best_params:
        print(f"Best Configuration: n_mfcc={best_params[0]}, delta={best_params[1]}, "
              f"norm={best_params[2]}, GMM_comp={best_params[3]}, "
              f"cov_type={best_params[4]}, mels={best_params[5]} -> Accuracy={best_accuracy:.4f}")