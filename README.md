# ODECE: Feasibility-Aware Decision-Focused Learning for Predicting Parameters in the Constraints

Code for the NeurIPS 2025 paper "Feasibility-Aware Decision-Focused Learning for Predicting Parameters in the Constraints"

## Summary

This repository contains the implementation of **ODECE** (optimizing decisions through end-to-end constraint estimation), a novel approach for Decision-Focused Learning (DFL) when predicting parameters that appear in the constraints of optimization problems.

We demonstrate the effectiveness of ODECE across multiple optimization problems:
- **Multidimensional Knapsack Problem (MDKP)**: Predicting item weights and knapsack capacities
- **Brass Alloy Blending**: Predicting element quantities in alloy composition

## Citation

If you use this code in your research, please cite our paper (will update after the proceedings are publicly available):

```bibtex
@misc{mandi2025feasibilityawaredecisionfocusedlearningpredicting,
      title={Feasibility-Aware Decision-Focused Learning for Predicting Parameters in the Constraints}, 
      author={Jayanta Mandi and Marianne Defresne and Senne Berden and Tias Guns},
      year={2025},
      eprint={2510.04951},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04951}, 
}
```

## Installation

We recommend using a virtual environment to avoid conflicts with other Python packages:

```bash
# Create a virtual environment
python3 -m venv env_dfl

# Activate the virtual environment
# On Linux/Mac
source env_dfl/bin/activate
# On Windows
# env_dfl\Scripts\activate

# Install dependencies
pip install -r requirement.txt
```

## Instructions for Running Experiments

To run all experiments, execute the master script from the command line:

```bash
bash run_all_exp.sh
```

This will sequentially run:
1. Brass Alloy experiments (`AlloyExp.sh`)
2. MDKP Weight prediction experiments (`MDKP_WeightExp.sh`)
3. MDKP Capacity prediction experiments (`MDKP_CapaExp.sh`)

Make sure the scripts have executable permissions:

```bash
chmod +x run_all_exp.sh AlloyExp.sh MDKP_WeightExp.sh MDKP_CapaExp.sh
```

### Running Individual Experiments

You can also run experiments individually:

```bash
# Brass Alloy experiments
bash AlloyExp.sh

# MDKP Weight prediction
bash MDKP_WeightExp.sh

# MDKP Capacity prediction
bash MDKP_CapaExp.sh
```

### Model Names in Experiment Scripts

The following models are implemented and compared in our experiments:

- **ODECE** (Ours): `odece` - optimizing decisions through end-to-end constraint estimation
- **Two-Stage Predict+Optimize**: `2sIntOpt` - Two-Stage Predict+Optimize for Mixed Integer Linear Programs with Unknown Parameters in Constraints. 
- **CombOptNet**: `comboptnet` - Combinatorial optimization network
- **Solver-free Learning**: `sfl` - Solver-Free Framework for Scalable Learning
- **MSE Baseline**: `mse` - Mean squared error baseline

Each model can be configured with various hyperparameters (learning rate, temperature, penalty coefficients, etc.) as specified in the experiment scripts.

## Processing Results

After running experiments, process the results using:

```bash
# For Alloy experiments
python3 read_result.py --result-dir Results/Alloy/

# For MDKP Weight experiments
python3 read_result.py --result-dir Results/KnapsackWeights/NoFixedCosts/

# For MDKP Capacity experiments
python3 read_result.py --result-dir Results/KnapsackCapacity/NoFixedCosts/
```

This will generate CSV files in the `CombinedResults/` directory containing aggregated metrics across all runs.

## Visualization

Generate scatter plots comparing model performance:

```bash
python3 visualization/MakeScatterPlot.py
```

This will create the scatter plot reported in the paper.

