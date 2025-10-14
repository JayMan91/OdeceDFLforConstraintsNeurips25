import numpy as np
import torch
from torch.autograd import Function
from torch import nn
from datetime import datetime
from OptProblems import opt
import pytorch_lightning as pl
import traceback
from src.pfl import PFL
from src.comboptnetsrc.comboptnet import CombOptNetModule
class CombOptNet(PFL):
    """
    A PyTorch Lightning module for Optimizing Decision Using Comboptnet https://github.com/martius-lab/CombOptNet

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
        dataset (Dataset): Training data if caching is used
        lr (float): Learning rate for optimizers
        max_epochs (int): Maximum number of training epochs
        scheduler: Learning rate scheduler
        seed (int): Random seed
    """
    def __init__(self, ml_predictor, optsolver, num_predconstrsvar,
        predict_indices=None, predict_cost=False, loss = 'regret', 
        tau= 0.5, lb=0, ub=1, num_warmup_epochs = 0,
        denormalize = False, poblem_name = 'knapsack', save_instance_wise_metrics=False,
        processes=1,  dataset=None, 
        lr=1e-3, max_epochs=100, scheduler=None, seed=135):

        super().__init__(ml_predictor, optsolver, num_predconstrsvar, predict_indices, predict_cost,
            denormalize, save_instance_wise_metrics, processes,  dataset, lr, max_epochs, scheduler, seed)
        
        self.solver_module = CombOptNetModule(dict(lb=lb, ub=ub), tau = tau)
        # self.automatic_optimization = True
        self.automatic_optimization = False
        self.loss = loss
        self.poblem_name = poblem_name
        self.save_hyperparameters('num_predconstrsvar', 'predict_indices', 'predict_cost', 'loss',
        'denormalize', 'max_epochs', 'num_warmup_epochs', 'lr', 'scheduler', 'seed', 'tau')

        # self.dflloss = dfl_method


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
                weights, capacity = pred_params_tuple
                # print ("Weights", weights[0])
                # print ("Capacity", capacity[0])
                # A @ y + b <= 0 is the format Comboptne expects
                ### Ref: https://github.com/martius-lab/CombOptNet/blob/d563d31a95dce35a365d50b81f932c27531ae09b/models/comboptnet.py#L104
                if self.poblem_name == 'knapsack':
                    a = weights
                    b = capacity * (-1)
                elif self.poblem_name == 'alloy':
                    a = weights * (-1)
                    b = capacity
                else:
                    raise ValueError("Invalid problem name")
                constraints = torch.cat((a, b.unsqueeze(-1)), dim=-1)
                # print ("Constraints", constraints[0])

                if self.optsolver.modelSense == opt.MAXIMIZE:
                    
                    costs_ = -pred_costs
                else:
                    costs_ = pred_costs

                pred_sols = self.solver_module(cost_vector = costs_, constraints = constraints)
                pred_sols = pred_sols.float()

                # pred_sols.retain_grad()
                # weights.retain_grad()

                if self.loss == 'mse':
                    criterion = nn.MSELoss()
                    total_loss = criterion(pred_sols, sols)
                elif self.loss == 'huber':
                    criterion = nn.HuberLoss()
                    total_loss = criterion(pred_sols, sols)
                elif self.loss == 'l1':
                    criterion = nn.L1Loss()
                    total_loss = criterion(pred_sols, sols)
                elif self.loss == 'regret': # 'decision loss'
                    total_loss = torch.einsum("bi,bi->b", costs_, pred_sols).mean()
                self.manual_backward(total_loss)
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                self.log('train_loss', total_loss, prog_bar=False, on_epoch=True, on_step=False)


        for predictor in self.constr_predictors.values():
            predictor.eval()
        if self.predict_cost:
            self.cost_predictor.eval()

    # def configure_optimizers(self):
    #     """
    #     Returns:
    #         list: List of optimizers in predict_indices order, with cost optimizer last if predict_cost is True.
    #              Always returns a list, even when there's only one optimizer.
    #     """
    #     optimizers = []

    #     # Create optimizers in the same order as predict_indices
    #     for idx in self.predict_indices:
    #         optimizer = torch.optim.Adam(self.constr_predictors[idx].parameters(), lr=self.hparams.lr)
    #         optimizers.append(optimizer)
        
    #     if self.predict_cost:
    #         optimizer = torch.optim.Adam(self.cost_predictor.parameters(), lr=self.hparams.lr)
    #         optimizers.append(optimizer)

    #     return optimizers