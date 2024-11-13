import os
import signal
import subprocess
import sys
from data_preprocessing import preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model

# File paths to check if processing has been done
processed_data_file = "processed_data.csv"
model_file = "best_model.pkl"
scaler_file = "scaler.pkl"
feature_name = "feature_names.pkl"
label_encoders = "label_encoders.pkl"

def main():
    # Check if preprocessing, training, and evaluation are already done
    if (os.path.exists(processed_data_file) and
        os.path.exists(model_file) and
        os.path.exists(scaler_file) and
        os.path.exists(feature_name) and
        os.path.exists(label_encoders)):
        print("Preprocessing, training, and evaluation already done. Starting deployment...")
    else:
        print("Starting preprocessing...")
        df = preprocess_data("NF-UQ-NIDS-v2.csv")
        if df is not None:
            df.to_csv(processed_data_file, index=False)
            print("Preprocessing completed.")

            print("Starting model training...")
            train_model(processed_data_file)
            print("Model training completed.")

            print("Starting model evaluation...")
            evaluate_model(processed_data_file)
            print("Model evaluation completed.")

    # Start the deployment server
    print("Starting deployment server...")
    deployment_process = subprocess.Popen(["python", "model_deployment.py"], cwd=os.getcwd())

    try:
        deployment_process.wait()
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Exiting gracefully...")
        deployment_process.terminate()
        deployment_process.wait()

def signal_handler(sig, frame):
    print("\nExiting gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    # Register the signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    main()