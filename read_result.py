import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import yaml
import argparse

def load_metrics(model_dir):
    """
    Load and aggregate metrics and hyperparameters in a directory.

    Args:
        model_dir (str): Path to the directory containing model version subdirectories.
                         Each version directory should contain a metrics.csv and hparams.yaml.

    Returns:
        list of pd.DataFrame: List of DataFrames, one per version, containing metrics and relevant hyperparameters.
    """
    # Initialize empty list to store DataFrames from each version
    all_dfs = []

    # Extract model name from directory path
    print ("Model directory ", model_dir)
    model_name = os.path.basename(model_dir).split('_')[0]  # Get first part before underscore
    print ("Model name ", model_name)

    # Iterate over all version directories
    for version in os.listdir(model_dir):
        print ("Model directory ", model_dir, "version: ", version)
        # Each subdirectory inside model_dir is a version directory containing experiment results
        version_path = os.path.join(model_dir, version)
        if os.path.isdir(version_path):
            metrics_file = os.path.join(version_path, "metrics.csv")
            hparams_file = os.path.join(version_path, "hparams.yaml")
            
            # Read hparams if available
            lr = None
            seed = None
            if os.path.exists(hparams_file):
                with open(hparams_file, 'r') as f:
                    hparams = yaml.safe_load(f)
                    lr = hparams.get('lr', None)
                    seed = hparams.get('seed', None)
                    denormalize = hparams.get('denormalize', None)
                    max_epochs = hparams.get('max_epochs', None)
                
                if "TwoStagePtO" in model_name:
                    thr = hparams.get('thr', None)
                    damping = hparams.get('damping', None)
                    knapsack_penalty = hparams.get('knapsack_penalty', 0.21)
                
                if "comboptnet" in model_name:
                    loss = hparams.get('loss', None)
                    tau = hparams.get('tau', None)
                    
                if "sfl" in model_name:
                    temp = hparams.get('temp', None)
                    
                if "odece" in model_name:
                    infeasibility_aversion_coeff = hparams.get('infeasibility_aversion_coeff', None)
                    normalize = hparams.get('normalize', None)
            
            if os.path.exists(metrics_file):
                try:
                    df = pd.read_csv(metrics_file)
                    
                    # Helper to safely get a column if it exists
                    get_col = lambda col: df[col] if col in df.columns else None
                    
                    # Create base metrics dictionary
                    metrics_dict = {
                        'epoch': get_col('epoch'),
                        'train_loss': get_col('train_loss'),
                        'val_infeasibility': get_col('val_infeasibility'),
                        'val_unsolvable': get_col('val_unsolvable'),
                        'val_regret': get_col('val_regret'),
                        'val_posthoc_regret': get_col('val_posthoc_regret'),
                        'val_recourse_cost': get_col('val_recourse_cost'),
                        'val_infeasible_regret': get_col('val_infeasible_regret'),
                        'val_is_true_feasible': get_col('val_is_true_feasible'),
                        'test_infeasibility': get_col('test_infeasibility'),
                        'test_unsolvable': get_col('test_unsolvable'),
                        'test_regret': get_col('test_regret'),
                        'test_posthoc_regret': get_col('test_posthoc_regret'),
                        'test_recourse_cost': get_col('test_recourse_cost'),
                        'test_infeasible_regret': get_col('test_infeasible_regret'),
                        'test_is_true_feasible': get_col('test_is_true_feasible'),
                        'lr': lr,
                        'max_epochs': max_epochs,
                        'seed': seed,
                        'denormalize': denormalize,
                        'model': model_name
                    }
                    
                    # Add all constraint loss columns
                    constraint_cols = [col for col in df.columns if col.startswith('val_loss_constraint_')]
                    for col in constraint_cols:
                        metrics_dict[col] = df[col]
                    
                    # Create DataFrame from dictionary
                    metrics = pd.DataFrame(metrics_dict)
                    
                    if "odece" in model_name:
                        
                        metrics['infeasibility_aversion_coeff'] = infeasibility_aversion_coeff
                        metrics['normalize'] = normalize
                    
                    if "TwoStagePtO" in model_name:
                        metrics['thr'] = thr
                        metrics['damping'] = damping
                        metrics['knapsack_penalty'] = knapsack_penalty
                    if "comboptnet" in model_name:
                        metrics['loss'] = loss
                        metrics['tau'] = tau
                        
                    if "sfl" in model_name:
                        metrics['temp'] = temp
                    
                    print ("Number of rows:", len(metrics))
                    all_dfs.append(metrics)
                except Exception as e:
                    print ("Error reading metrics file:", metrics_file)
                    print (str(e))

    # Concatenate all DataFrames
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze results from experiment runs')
    parser.add_argument('--result-dir', type=str, required=True,
                        help='Base directory containing the results')
    # for example: python3 read_result.py --result-dir Results/Alloy/
    args = parser.parse_args()

    # Normalize base directory path by removing trailing slashes
    base_dir = args.result_dir.rstrip('/')

    # Load data for both models
    all_results = []
    # Iterate over all model directories
    for model_dir in os.listdir(base_dir):
        full_path = os.path.join(base_dir, model_dir)
        if os.path.isdir(full_path):
            df = load_metrics(full_path)
            if not df.empty:
                # Extract parameters from directory name
                if "knapsack" in base_dir.lower():
                    match = re.match(r'.*_deg(\d+)_noise(\d+\.\d+)_numitems(\d+)', model_dir)
                    if match:
                        df['degree'] = int(match.group(1))
                        df['noise'] = float(match.group(2))
                        df['num_items'] = int(match.group(3))
                elif "alloy" in base_dir.lower():
                    match = re.match(r'.*_penalty_(\d+\.\d+)', model_dir)
                    if match:
                        df['penalty'] = float(match.group(1))
                all_results.append(df)

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        # Join the last two directory names for the output CSV name
        parts = base_dir.rstrip('/').split('/')
        csv_name = ''.join(parts[-2:])
        combined_df.to_csv(f"CombinedResults/{csv_name}.csv", index=False)

if __name__ == "__main__":
    main()