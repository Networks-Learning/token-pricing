import os
import pickle
import json
import sys
from typing import Any

def custom_serializer(obj: Any):
    """
    Custom function to serialize non-JSON-compatible objects.
    """
    try:
        # Handle PyTorch tensors
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  # Convert tensors to lists
    except ImportError:
        pass

    try:
        # Handle NumPy arrays
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert arrays to lists
    except ImportError:
        pass

    # Fallback for unsupported types
    return str(obj)

def convert_pickle_to_json(folder_path):
    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file has a .pkl extension
        if file_name.endswith('.pkl'):
            pickle_file_path = os.path.join(folder_path, file_name)
            json_file_path = os.path.join(folder_path, f"{os.path.splitext(file_name)[0]}.json")
            
            try:
                # Load the pickle file
                with open(pickle_file_path, 'rb') as pickle_file:
                    data = pickle.load(pickle_file)
                
                # Save the data to a JSON file
                with open(json_file_path, 'w') as json_file:
                    json.dump(data, json_file, indent=4, default=custom_serializer)
                
                print(f"Converted {file_name} to {os.path.basename(json_file_path)}")
            except Exception as e:
                print(f"Failed to convert {file_name}: {e}")

if __name__ == "__main__":
    # Get the current directory
    folder_path = os.getcwd()
    print(f"Converting pickle files in folder: {folder_path}")
    convert_pickle_to_json(folder_path)