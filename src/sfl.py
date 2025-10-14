import numpy as np
import torch
from torch.autograd import Function
from torch import nn
from datetime import datetime
from OptProblems import opt
import pytorch_lightning as pl
import traceback
from src.pfl import PFL
from src.sflsrc.losses import ILPLoss, CoVBalancer
class SFL(PFL):
    """
    A PyTorch Lightning module for Optimizing Decision Using solver-free ILP–Loss (SFL) https://github.com/dair-iitd/ilploss/tree/main/src
    Their problem formulation is the form of:

    min c^T x s.t. A x + b >= 0

    Args:   
        ml_predictor (list): List of nn.Module(s) to predict parameters. Models are mapped
                            to predict_indices in order, e.g., if predict_indices=[2,0,5], then
                            ml_predictor[0] predicts var[2], ml_predictor[1] predicts var[0], etc.

                            Note: The order of optimizers will match the sorted predict_indices,
                            not the order in ml_predictor. For example, if predict_indices=[2,0,5],
                            then optimizer[0] is for variable 0, optimizer[1] for variable 2, etc.
        optsolver (OptSolver): Optimization solver instance
        num_predconstrsvar (int): Total number of variables in the constraints
        predict_indices (list, optional): Indices of variables to predict. Must be within [0, num_predconstrsvar).
                                        If None, predicts all variables up to num_predconstrsvar.
                                        Will be sorted to ensure consistent ordering.
        predict_cost (bool): Whether to predict the cost parameter. If True, last predictor in ml_predictor
                           is used for cost prediction.
        loss (str, optional): Type of loss function to use. Options are 'mse', 'l1' 'huber' and 'decision'. Default is decision loss
        processes (int): Number of processors
        poblem_name (str): Name of the problem, Rquires to determing the constraints
        dataset (Dataset): Training data if caching is used
        lr (float): Learning rate for optimizers
        max_epochs (int): Maximum number of training epochs
        scheduler: Learning rate scheduler
        seed (int): Random seed
    """
    def __init__(self, ml_predictor, optsolver, num_predconstrsvar, sampler : nn.Module,
        predict_indices=None, predict_cost=False, loss = None, temp = 0.5, num_warmup_epochs = 0,
        denormalize = False, poblem_name = 'knapsack',save_instance_wise_metrics=False, processes=1,  dataset=None, 
        lr=1e-3, max_epochs=100, scheduler=None, seed=135):

        super().__init__(ml_predictor, optsolver, num_predconstrsvar, predict_indices, predict_cost,
            denormalize, save_instance_wise_metrics, processes,  dataset, lr, max_epochs, scheduler, seed)
        self.save_hyperparameters('num_predconstrsvar', 'predict_indices', 'predict_cost', 
        'denormalize', 'max_epochs', 'num_warmup_epochs', 'lr', 'scheduler', 'seed', 'temp')
        
        
        # self.automatic_optimization = True
        self.automatic_optimization = False
        self.balancer = CoVBalancer(num_losses = 3)
        self.criterion = ILPLoss (self.balancer, pos_param = 0.01, neg_param = 0.01, temp = temp)
        self.sampler = sampler
        self.poblem_name = poblem_name




    def training_step(self, batch, batch_idx):
        # Get optimizer and ensure it's a list
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        

        for predictor in self.constr_predictors.values():
            predictor.train()
        if self.predict_cost:
            self.cost_predictor.train()
        features, trueConstr_tuple, costs, sols, objs, penalty = batch
        # print ("Inputs Shape: ", costs.shape, sols.shape, objs.shape)

        if self.hparams.num_warmup_epochs > 0 and self.current_epoch < self.hparams.num_warmup_epochs:
            criterion = nn.MSELoss()

            # Train variable predictors
            for i, predictor in self.constr_predictors.items():
                predicted_params = predictor(features)
                
                # Access optimizer using idx_to_opt_pos to map from variable index to optimizer position
                # For example, if predict_indices=[2,0,5], then idx_to_opt_pos[2]=1, so we use optimizers[1]
                optimizer = optimizers[self.idx_to_opt_pos[i]]
                
                optimizer.zero_grad()
                loss = criterion(predicted_params, trueConstr_tuple[i])
                self.manual_backward(loss)
                optimizer.step()
                self.log(f'train_loss_constraint_{i}', loss, prog_bar=True,  on_epoch=True, on_step=False)
        
        else:

            pos = sols

            for i, predictor in self.constr_predictors.items():
                # Access optimizer using idx_to_opt_pos to map from variable index to optimizer position
                # For example, if predict_indices=[2,0,5], then idx_to_opt_pos[2]=1, so we use optimizers[1]
                optimizer = optimizers[self.idx_to_opt_pos[i]]
                model = self.constr_predictors[i]  
                optimizer.zero_grad()

                predicted_params = self.forward(features)
                pred_params_tuple, pred_costs = self._create_params_tuple(predicted_params,
                                                                        trueConstr_tuple,
                                                                        costs)
                ##  #(a, y_obj, GRB.GREATER_EQUAL, -b) is their model i.e. ax >= -b => ax + b >= 0
                ## Reference:https://github.com/dair-iitd/ilploss/blob/19bb10ef987f64f86e256401efda5ce47a5b0b12/src/CombOptNet/src/ilp.py#L224C12-L224C58
                if self.poblem_name == 'knapsack':
                    weights, capacity = pred_params_tuple
                    a_l = weights*(-1)
                    b_l = capacity
                    a_k = torch.zeros_like(weights[:, :0])
                    b_k = torch.zeros_like(capacity[:, :0])
                elif self.poblem_name == 'alloy':
                    weights, req = pred_params_tuple
                    a_l = weights
                    b_l = req * (-1)
                    a_k = torch.zeros_like(weights[:, :0])
                    b_k = torch.zeros_like(req[:, :0])
                else:
                    raise ValueError("Invalid problem name")

                if self.optsolver.modelSense == opt.MAXIMIZE:
                    costs_ = -pred_costs
                else:
                    costs_ = pred_costs

                with torch.no_grad():
                    neg = self.sampler([a_l, a_k], [b_l, b_k], 
                                    costs_,   sols)
                
                # loss = criterion(predicted_params, trueConstr_tuple[i])
                solverfree_loss = self.criterion ([a_l, a_k], 
                                            [b_l, b_k], 
                                            costs_, pos, neg)
                self.manual_backward(solverfree_loss)
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                self.log('train_loss', solverfree_loss, prog_bar=False, on_epoch=True, on_step=False)

        for predictor in self.constr_predictors.values():
            predictor.eval()
        if self.predict_cost:
            self.cost_predictor.eval()

        
    def on_train_epoch_end(self) :
        
        if self.current_epoch in [9, 29]:
            self.criterion.temp *= 0.2
