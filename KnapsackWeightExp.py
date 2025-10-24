import argparse
from argparse import Namespace
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from OptProblems.knapsack.kpdata import genCapacity, genWeights
from OptProblems.knapsack.kpdataset import knapsack_dataset
from OptProblems.knapsack.kpsolver import knapsack_solver
from src.pfl import PFL
from src.odece import ODECE
from src.sfl import SFL
from src.comboptnet import CombOptNet
from src.TwoStage import TwoStage
import numpy as np
import os   
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, help="name of models", default="", required= True)
parser.add_argument("--num_items", type=int, help="number of knapsack items", default= 100, required= False)
parser.add_argument("--num_data", type= int, help="Number of Training Data", default= 1000, required= False)
parser.add_argument("--num_feat", type= int, help="Number of Features", default= 10, required= False)
parser.add_argument("--dim", type=int, help="dimension of knapsack", default= 3, required= False)
parser.add_argument("--deg", type=int, help="degree of misspecifaction", default= 4, required= False)
parser.add_argument("--noise_width", type=float, help="noise width misspecifaction", default= 0.1, required= False)

# parser.add_argument("--capacity_ratio", type=float, help="ratio of capacity", default= 0.5, required= False)
parser.add_argument("--seed", type=int, help="random seed", default= 135, required= False)
parser.add_argument("--batch_size", type=int, help="batch size", default= 32, required= False)
parser.add_argument("--max_epochs", type=int, help="max epochs", default= 20, required= False)
parser.add_argument("--lr", type=float, help="Learning rate", default= 5e-3, required= False)
parser.add_argument("--denormalize", type=bool, help="denormalize", default= False, required= False)
### ODECE Ablation
parser.add_argument("--infeasibility_aversion_coeff", type=float, help="infeasibility_aversion_coeff", default= 0.5, required= False)
parser.add_argument("--margin_threshold", type=float, help="margin_threshold", default= 2., required= False)
# Comboptnet parameters
parser.add_argument("--loss", type=str, help="loss", default= 'l1', required= False)
parser.add_argument("--tau", type=float, help="tau", default= 0.5, required= False)
parser.add_argument("--penalty", type=float, help="penalty", default= 0.21, required= False)
### SFL parameters
parser.add_argument("--temp", type=float, help="temp", default= 0.5, required= False)
### TwoStage parameters
parser.add_argument("--thr", type=float, help="thr", default= 0.1, required= False)
parser.add_argument("--damping", type=float, help="damping", default= 0.01, required= False)
args = parser.parse_args()

argument_dict = vars(args)
num_items, num_data, num_feat, deg, noise_width, dim = argument_dict['num_items'], \
    argument_dict['num_data'], argument_dict['num_feat'], argument_dict['deg'], argument_dict['noise_width'], argument_dict['dim']
seed =  argument_dict['seed']
batch_size = argument_dict ['batch_size']
max_epochs = argument_dict['max_epochs']
lr = argument_dict['lr']
denormalize = argument_dict['denormalize']
infeasibility_aversion_coeff = argument_dict['infeasibility_aversion_coeff']
margin_threshold = argument_dict['margin_threshold']
# Comboptnet parameters
loss = argument_dict['loss']
tau = argument_dict['tau']
penalty = argument_dict['penalty']
# SFL parameters
temp = argument_dict['temp']
### TwoStage parameters
thr = argument_dict['thr']
damping = argument_dict['damping']
# generate data
num_test_instances = 500
capacity_ratio  = np.array ([0.18, 0.2, 0.22])
X, w, costs, capacity = genWeights(num_data + num_test_instances, num_feat, num_items,
 capacity_ratio, dim, deg, noise_width, seed = seed)
print ("Data Generated")
print (capacity[0])
print ("weights", w.max())

x_train, x_test, w_train, w_test = train_test_split(X, w, test_size=num_test_instances, random_state=seed)
x_train, x_val, w_train, w_val  = train_test_split(x_train, w_train, test_size=0.1, random_state=seed) 

# get optDataset
dataset_train = knapsack_dataset(x_train, w_train, capacity, costs, num_items)
dataset_val = knapsack_dataset(x_val, w_val, capacity, costs, num_items)
dataset_test = knapsack_dataset(x_test, w_test, capacity, costs, num_items)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=19, persistent_workers=True)
loader_val  = DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=19, persistent_workers=True)
loader_test = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=19, persistent_workers=True)

from ML.TorchML import LinearRegressionforKP_PredWeight , LinearRegressionforKP_NormalizedPredWeight

solver = knapsack_solver(num_items)
reg = LinearRegressionforKP_PredWeight( num_feat, num_items, dim) # init model


log_dir = os.getcwd() + "/Results/KnapsackWeights/NoFixedCosts/"
if argument_dict['model_name'] == 'odece':
    logger = CSVLogger(
        log_dir, 
        name='odece_deg{}_noise{}_numitems{}'.format(deg, noise_width, num_items)
    )
    model = ODECE(
        [reg], 
        optsolver=solver, 
        num_predconstrsvar=1, 
        denormalize=denormalize,
        infeasibility_aversion_coeff = infeasibility_aversion_coeff,
        margin_threshold = margin_threshold,
        predict_cost=False, 
        lr=lr, 
        max_epochs=max_epochs,
        seed=seed
    )

elif argument_dict['model_name'] == 'mse':
    logger = CSVLogger(
        log_dir, 
        name='mse_deg{}_noise{}_numitems{}'.format(deg, noise_width, num_items)
    )
    model = PFL(
        [reg], 
        optsolver=solver, 
        num_predconstrsvar=1, 
        denormalize=denormalize,
        predict_cost=False, 
        lr=lr, 
        max_epochs=max_epochs,
        seed=seed
    )
elif argument_dict['model_name'] == 'sfl':
    from src.sflsrc.samplers import DefaultSamplerList
    logger = CSVLogger(
        log_dir, 
        name='sfl_deg{}_noise{}_numitems{}'.format(deg, noise_width, num_items)
    )
    model = SFL(
        [reg], 
        sampler=DefaultSamplerList,
        optsolver=solver, 
        num_predconstrsvar=1, 
        poblem_name = 'knapsack',
        denormalize=denormalize,
        predict_cost=False, 
        temp = temp,
        lr=lr, 
        max_epochs=max_epochs,
        seed=seed
    )
elif argument_dict['model_name'] == 'comboptnet':
    logger = CSVLogger(
        log_dir, 
        name='comboptnet_deg{}_noise{}_numitems{}'.format(deg, noise_width, num_items)
    )
    model = CombOptNet(
        [reg], 
        optsolver=solver, 
        num_predconstrsvar=1, 
        poblem_name = 'knapsack',
        denormalize=denormalize,
        predict_cost=False, 
        loss = loss,
        tau = tau,
        lr=lr, 
        max_epochs=max_epochs,
        seed=seed
    )
elif argument_dict['model_name'] == 'TwoStagePtO':
    logger = CSVLogger(
        log_dir, 
        name= 'TwoStagePtO_deg{}_noise{}_numitems{}'.format(deg, noise_width, num_items)
    )
    model = TwoStage(
        [reg],
        optsolver=solver, 
        num_predconstrsvar=1, 
        poblem_name = 'knapsack',   
        denormalize=False, 
        predict_cost=False, 
        thr = thr,
        damping = damping,
        knapsack_penalty = penalty,
        lr=lr, 
        max_epochs=max_epochs,
        seed=seed
    )
else:
    raise ValueError("Invalid model name")

print ("="*100, "\n")
print ("We will training model: ", argument_dict['model_name'])
print ("="*100, "\n")
trainer = pl.Trainer(max_epochs= max_epochs, check_val_every_n_epoch=1, logger = logger)
trainer.validate(model, dataloaders = loader_val )
trainer.fit(model,  train_dataloaders= loader_train, val_dataloaders= loader_val)
print("Test Result: ", trainer.test(dataloaders = loader_test) )