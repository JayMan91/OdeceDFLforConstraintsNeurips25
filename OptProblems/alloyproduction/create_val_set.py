import numpy as np
import os

def read_and_split_files(base_path, file_prefix, num_files=1, is_penalty=False, penalty_term=0.5):
    all_data = []
    for testi in range(1):
        if is_penalty:
            # For penalty files, combine into a single directory name
            penalty_dir = f"test_penalty{penalty_term}"
            filename = os.path.join(base_path, penalty_dir, file_prefix + "(" + str(testi) + ").txt")
        else:
            filename = os.path.join(base_path, file_prefix + "(" + str(testi) + ").txt")
        print("Reading file:", filename)  # Debug print
        data = np.loadtxt(filename)
        all_data.append(data)
    
    # Stack all data
    all_data = np.vstack(all_data)
    
    # Calculate split point (1/3)
    split_idx = len(all_data) // 3
    
    # Split into validation and test
    val_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    return val_data, test_data

def main():
    # Base paths
    base_path = "/home/jayanta/Documents/Projects/CELPiC/data/Alloy production/brass"
    test_prices_path = os.path.join(base_path, "test_prices")
    test_features_path = os.path.join(base_path, "test_features")
    test_weights_path = os.path.join(base_path, "test_weights")
    test_penalty_path = os.path.join(base_path)
    
    # Process each type of data
    val_prices, test_prices = read_and_split_files(test_prices_path, "test_prices")
    val_features, test_features = read_and_split_files(test_features_path, "test_features")
    val_weights, test_weights = read_and_split_files(test_weights_path, "test_weights")
    val_penalty, test_penalty = read_and_split_files(test_penalty_path, "test_penalty", is_penalty=True, penalty_term=0.25)
    
    # Create validation directory if it doesn't exist
    val_dir = os.path.join(base_path, "validation")
    os.makedirs(val_dir, exist_ok=True)
    
    # Save validation data
    np.savetxt(os.path.join(val_dir, "val_prices.txt"), val_prices)
    np.savetxt(os.path.join(val_dir, "val_features.txt"), val_features)
    np.savetxt(os.path.join(val_dir, "val_weights.txt"), val_weights)
    np.savetxt(os.path.join(val_dir, "val_penalty.txt"), val_penalty)
    
    # Save new test data (overwrite existing)
    np.savetxt(os.path.join(base_path, "test_prices_new.txt"), test_prices)
    np.savetxt(os.path.join(base_path, "test_features_new.txt"), test_features)
    np.savetxt(os.path.join(base_path, "test_weights_new.txt"), test_weights)
    np.savetxt(os.path.join(base_path, "test_penalty_new.txt"), test_penalty)
    
    print("Validation and test sets created successfully!")

if __name__ == "__main__":
    main()
