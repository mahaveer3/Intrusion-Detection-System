# Intrusion Detection System (IDS) Project

## Overview
This project is an Intrusion Detection System (IDS) that uses machine learning techniques to detect anomalies in network traffic. The system is built using Python and includes data preprocessing, model training, evaluation, and deployment.

## Project Structure
The project is organized into the following files and directories:

- `data_preprocessing.py`: Script for preprocessing the data.
- `main.py`: Main script to run the IDS.
- `model_training.py`: Script for training the machine learning model.
- `model_evaluation.py`: Script for evaluating the trained model.
- `model_deployment.py`: Script for deploying the model.
- `view.py`: Script for the Flask web application.
- `scaler.pkl`: Saved scaler model.
- `feature_names.pkl`: Saved feature names.
- `label_encoders.pkl`: Saved label encoders.
- `processed_data.csv`: Preprocessed data.
- `NF-UQ-NIDS-v2.csv`: Original dataset.

## Setup Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/Mahaveer-G8_CS/Final-Capstone-Project.git
    ```
2. Navigate to the project directory
    ```bash
    cd /path/to/IDS_PROJECT
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the main script:
    ```bash
    chmod +x main.py

    python main.py
    ```

## Usage
- **Data Preprocessing**: The `data_preprocessing.py` script reads the raw data, processes it, and saves the processed data.
    ```python
    def preprocess_data(file_path):
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print("The specified file was not found.")
            return None
    ```

- **Model Training**: The `model_training.py` script trains the machine learning model using the preprocessed data.
    ```python
    def train_model(processed_data_file):
        # Training logic here
    ```

- **Model Evaluation**: The `model_evaluation.py` script evaluates the performance of the trained model.
    ```python
    def evaluate_model(model_file, test_data_file):
        # Evaluation logic here
    ```

- **Model Deployment**: The `model_deployment.py` script deploys the trained model for real-time predictions.
    ```python
    def deploy_model(model_file):
        # Deployment logic here
    ```

- **Flask Web Application**: The `view.py` script runs a Flask web application for interacting with the IDS.
    ```python
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        # Prediction logic here
    ```

## Troubleshooting
- Ensure all necessary files are present before running the scripts.
    ```python
    if (os.path.exists(processed_data_file) and
        os.path.exists(model_file) and
        os.path.exists(scaler_file) and
        os.path.exists(feature_name) and
        os.path.exists(label_encoders)):
        print("Preprocessing, training, and evaluation already done. Starting deployment...")
    else:
        print("Starting preprocessing...")
    ```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.
