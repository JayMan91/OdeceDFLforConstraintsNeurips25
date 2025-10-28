#!/bin/bash

# Script to process all experiment results

echo "Processing Alloy experiment results..."
python3 read_result.py --result-dir Results/Alloy/

echo "Processing MDKP Weight experiment results..."
python3 read_result.py --result-dir Results/KnapsackWeights/NoFixedCosts/

echo "Processing MDKP Capacity experiment results..."
python3 read_result.py --result-dir Results/KnapsackCapacity/NoFixedCosts/

echo "Combined results saved in CombinedResults/ directory"
