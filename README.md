# GMM Spoken Digit Recognition 

A complete pipeline for recognizing spoken digits (0-9) using **Mel-Frequency Cepstral Coefficients (MFCC)** and **Gaussian Mixture Models (GMM)**.

## Overview

Originally developed as an academic group project for the *Speech Technology* (Technologia Mowy) course at AGH University, this repository has been refactored to meet modern software engineering standards. It demonstrates a classic, statistical DSP approach to audio classification without relying on deep learning algorithms.

## Key Features

* **Advanced Feature Extraction:** Extracts MFCCs, applies Cepstral Mean and Variance Normalization (CMVN), and computes Delta/Delta-Delta coefficients.
* **Robust Validation:** Uses a speaker-aware Stratified K-Fold cross-validation to ensure the model learns phonetic features rather than simply memorizing individual speaker voices.
* **Hyperparameter Tuning:** Includes an automated grid search pipeline to find the optimal combination of GMM components, Mel filters, and normalization modes.


## Getting Started

### Prerequisites

Ensure you have Python installed along with the necessary data science libraries:
```bash

pip install numpy librosa scikit-learn matplotlib

```
## Dataset Information

The audio dataset used for this project was provided as a closed academic resource by AGH University and is not included in this public repository.

To run this pipeline locally, you can use your own `.wav` files *(or a public dataset such as the Free Spoken Digit Dataset)* structured in the root directory as follows:

```text
digits/
├── train/   # files for training
└── eval/    # files for evaluation
```

## Running the Pipeline

To run the complete process *(data grouping, model training, prediction generation, and evaluation)*, simply execute the main entry point:

```bash

python main.py

```

## Baseline Results
The current baseline model achieves **71.00% accuracy** on the unseen evaluation dataset.

- **Strongest predictions:** Digits **2** and **8** are recognized with near-perfect accuracy.

- **Known limitations:** The statistical GMM architecture occasionally confuses phonetically similar numbers containing the **/aɪ/** diphthong *(e.g., 5 and 9)*.

## Acknowledgments
- **Original R&D Team:** Kacper Góral, Damian Pająk, Igor Kowalski
- **Evaluation Script:** Provided by Marcin Witkowski *(AGH TM Lab)*
- **Original Repository:** [Link to the original project](https://github.com/skynni/TM_projekt_1)