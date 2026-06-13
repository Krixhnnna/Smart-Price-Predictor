#!/usr/bin/env python3
"""
save_model.py — Helper script to export the trained scikit-learn model
pipeline from the local ZenML store and serialize it as a joblib file.
"""

import os
import sys
import joblib
from zenml.client import Client

def main():
    print("Initializing ZenML client...")
    client = Client()
    
    model_name = "prices_predictor"
    
    try:
        print(f"Retrieving latest model version for '{model_name}' from ZenML registry...")
        # Get the latest model version directly
        model_version = client.get_model_version(model_name)
        print(f"Using model version: {model_version.name} (version ID: {model_version.id})")
        
        # Get the sklearn_pipeline artifact
        print("Loading artifact 'sklearn_pipeline'...")
        artifact = model_version.get_model_artifact("sklearn_pipeline")
        
        # Load the pipeline object in-memory
        pipeline = artifact.load()
        
        # Save to local file
        output_filename = "model.joblib"
        print(f"Serializing pipeline to '{output_filename}'...")
        joblib.dump(pipeline, output_filename)
        
        print(f"\nSuccess! Saved model to {os.path.abspath(output_filename)}")
        print(f"File size: {os.path.getsize(output_filename) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to export model: {e}", file=sys.stderr)
        print("\nPlease ensure that you have run the training pipeline first by running:", file=sys.stderr)
        print("  python run_pipeline.py", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
