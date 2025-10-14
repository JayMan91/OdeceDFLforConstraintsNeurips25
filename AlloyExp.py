import argparse
from argparse import Namespace
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from OptProblems.alloyproduction.alloydataset import alloy_dataset
from OptProblems.alloyproduction.alloysolver import alloy_solver
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
parser.add_argument("--penaltyTerm", type=float, help="penalty term", default= 0.25, required= False)
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
### SFL parameters
parser.add_argument("--temp", type=float, help="temp", default= 0.5, required= False)
### Twostaeg parameters
parser.add_argument("--thr", type=float, help="thr", default= 0.1, required= False)
parser.add_argument("--damping", type=float, help="damping", default= 0.01, required= False)
args = parser.parse_args()
argument_dict = vars(args)
seed =  argument_dict['seed']
batch_size = argument_dict ['batch_size']
max_epochs = argument_dict['max_epochs']
lr = argument_dict['lr']
penaltyTerm = argument_dict['penaltyTerm']
denormalize = argument_dict['denormalize']
infeasibility_aversion_coeff = argument_dict['infeasibility_aversion_coeff']
margin_threshold = argument_dict['margin_threshold']
# Comboptnet parameters
loss = argument_dict['loss']
tau = argument_dict['tau']
### SFL parameters
temp = argument_dict['temp']
### TwoStage parameters
thr = argument_dict['thr']
damping = argument_dict['damping']
# get optDataset
dataset_train = alloy_dataset(mode = 'train', penaltyTerm = penaltyTerm)
dataset_val = alloy_dataset(mode = 'val', penaltyTerm = penaltyTerm)
dataset_test = alloy_dataset(mode = 'test', penaltyTerm = penaltyTerm)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=19)
loader_val  = DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=19)
loader_test = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=19)

num_items = 10
num_feat = 4096
dim = 10

solver = alloy_solver(num_items)


from ML.TorchML import LinearRegressionforKP_PredCapacity , AlloyNet
reg = AlloyNet(num_layers = 1, num_features = num_feat, num_targets = 1) # init model
log_dir = os.getcwd() + "/Results/Alloy/"


if argument_dict['model_name'] == 'odece':
    logger = CSVLogger(
        log_dir, 
        name= f'odece_penalty_{penaltyTerm}'
    )
    model = ODECE(
        [reg],
        optsolver=solver, 
        num_predconstrsvar=1, 
        denormalize=denormalize, 
        predict_cost=False, 
        infeasibility_aversion_coeff = infeasibility_aversion_coeff,
        margin_threshold = margin_threshold,
        lr=lr, 
        max_epochs=max_epochs,
        seed=seed
    )



elif argument_dict['model_name'] == 'mse':
    logger = CSVLogger(
        log_dir, 
        name= f'mse_penalty_{penaltyTerm}'
    )
    model = PFL(
        [reg],
        optsolver=solver, 
        num_predconstrsvar=1, 
        denormalize=False, 
        predict_cost=False, 
        lr=lr, 
        max_epochs=max_epochs,
        seed=seed
    )

elif argument_dict['model_name'] == 'sfl':
    from src.sflsrc.samplers import DefaultSamplerList
    logger = CSVLogger(
        log_dir, 
        name= f'sfl_penalty_{penaltyTerm}'
    )
    model = SFL(
        [reg],
        sampler=DefaultSamplerList,
        optsolver=solver, 
        num_predconstrsvar=1, 
        poblem_name = 'alloy',
        denormalize=False,
        predict_cost=False, 
        lr=lr, 
        max_epochs=max_epochs,
        temp = temp,
        seed=seed
    )

elif argument_dict['model_name'] == 'comboptnet':
    logger = CSVLogger(
        log_dir, 
        name= f'comboptnet_penalty_{penaltyTerm}'
    )
    model = CombOptNet(
        [reg],
        optsolver=solver, 
        num_predconstrsvar=1, 
        poblem_name = 'alloy',
        lb = 0,
        ub = 1e+5,
        denormalize=False, 
        predict_cost=False, 
        loss = loss,
        tau = tau,
        lr=lr, 
        max_epochs=max_epochs,
        seed=seed
    )
elif argument_dict['model_name'] == 'TwoStageIntOpt':
    logger = CSVLogger(
        log_dir, 
        name= f'TwoStageIntOpt_penalty_{penaltyTerm}'
    )
    model = TwoStage(
        [reg],
        optsolver=solver, 
        num_predconstrsvar=1, 
        poblem_name = 'alloy',  
        denormalize=False, 
        predict_cost=False, 
        thr = thr,
        damping = damping,
        lr=lr, 
        max_epochs=max_epochs,
        seed=seed
    )
else:
    raise ValueError("Invalid model name")

print ("="*100, "\n")
print ("We will training model: ", argument_dict['model_name'])
print ("="*100, "\n")
trainer = pl.Trainer(max_epochs= max_epochs, check_val_every_n_epoch=1,  logger = logger)
trainer.validate(model, dataloaders = loader_val )
trainer.fit(model,  train_dataloaders= loader_train, val_dataloaders= loader_val)
print("Test Result: ", trainer.test(dataloaders = loader_test) )
