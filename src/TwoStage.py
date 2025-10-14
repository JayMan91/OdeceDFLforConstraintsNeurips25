import numpy as np
import torch
from torch.autograd import Function
from torch import nn
from datetime import datetime
from OptProblems import opt
import pytorch_lightning as pl
import traceback
from src.pfl import PFL
from src.TwoStagesrc.ip_model_whole import IPOfunc
import src.TwoStagesrc.ip_model_whole as ip_model_wholeFile
from src.TwoStagesrc.ip_model_whole_knapsack import IPOfuncKnapsack
import src.TwoStagesrc.ip_model_whole_knapsack as ip_model_wholeFile_knapsack

class TwoStage(PFL):
    def __init__(self, ml_predictor, optsolver, num_predconstrsvar,
        predict_indices=None, predict_cost=False, 
        max_iter=None, thr=0.1, damping=0.5, smoothing=False,
        denormalize = False,num_warmup_epochs =1 , knapsack_penalty = 0.21,
        poblem_name = 'knapsack',save_instance_wise_metrics=False,
        processes=1,  dataset=None, 
        lr=1e-3, max_epochs=100, scheduler=None, seed=135):
        super().__init__(ml_predictor, optsolver, num_predconstrsvar, predict_indices, predict_cost,
            denormalize, save_instance_wise_metrics, processes,  dataset, lr, max_epochs, scheduler, seed)
        
        self.automatic_optimization = False
        self.max_iter = max_iter
        self.problem_name = poblem_name

        self.smoothing = smoothing
        self.save_hyperparameters('num_predconstrsvar', 'predict_indices', 'predict_cost', 
        'denormalize', 'max_epochs', 'num_warmup_epochs', 'lr', 'scheduler', 'seed', 'max_iter',
         'thr', 'damping', 'smoothing', 'knapsack_penalty')
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
        batch_size = len(features)
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
            for i, predictor in self.constr_predictors.items():
                optimizer = optimizers[self.idx_to_opt_pos[i]]
                model = self.constr_predictors[i]  
                optimizer.zero_grad()

                predicted_params = self.forward(features)
                pred_params_tuple, pred_costs = self._create_params_tuple(predicted_params,
                                                                        trueConstr_tuple,
                                                                        costs)
                if self.optsolver.modelSense == opt.MAXIMIZE:
                    costs_ = -pred_costs
                else:
                    costs_ = pred_costs 
        
                weights, capacity = pred_params_tuple
                true_weights, true_capacity = trueConstr_tuple

                # print ("Shape Check")
                # print ("Capacity shape: ", capacity.shape)
                # print ("Cost shape: ", costs_.shape)
                # print ("Weights shape: ", weights.shape)
                # print ("True weights shape: ", true_weights.shape)
                # print  ("penalty shape: ", penalty.shape)
                batch_loss = 0

                for batch in range(batch_size):
                    if self.problem_name == 'alloy':

                        h_batch = -capacity[batch]
                        c_batch = costs_[batch]
                        G_batch=  -true_weights[batch]
                        penalty_batch = penalty[batch]
                        pred_G = -weights[batch]
                        A = torch.zeros (2, weights.shape[-1]).float()
                        b = torch.zeros (2).float() 

                        x_s2 = IPOfunc(A=A, b=b,  h = h_batch, c= c_batch, GTrue=G_batch,
                        penalty=penalty_batch, max_iter=self.max_iter, 
                        thr=self.hparams.thr, damping=self.hparams.damping,
                        smoothing=self.smoothing)(pred_G)
                        x_s1 = ip_model_wholeFile.x_s1
                        # print ("x_s1",x_s1.shape)
                        # print ("x_s2",x_s2.shape)
            
                        loss = torch.dot( penalty_batch * c_batch, abs(x_s2-x_s1).float()) + (x_s2 * c_batch).sum()
                        batch_loss += loss

                    elif self.problem_name == 'knapsack':
                        # compensation_fee, purchase_fee = 0.21, self.hparams.knapsack_penalty
                        compensation_fee, purchase_fee = 0.01, 8

                        n_items = true_weights.shape[-1]
                        h_batch = torch.cat([torch.ones(n_items, device=capacity.device), capacity[batch]])
                        c_batch = -costs_[batch]
                        G_batch = torch.cat([torch.eye(n_items, device=true_weights.device), 
                            true_weights[batch]], dim=0)
                        penalty_batch = penalty[batch]
                        pred_G = weights[batch]
                        A = torch.zeros (2, weights.shape[-1]).float()
                        b = torch.zeros (2).float()  
                        try:  

                            x_s2 = IPOfuncKnapsack(A=A, b=b, h=h_batch, c=c_batch, GTrue=G_batch,
                            purchase_fee= purchase_fee, compensation_fee=compensation_fee,
                            dim_kp= len(true_weights[batch]),
                            max_iter=self.max_iter, thr=self.hparams.thr, damping=self.hparams.damping,
                            smoothing=self.smoothing)(pred_G)    
                            x_s1 = ip_model_wholeFile_knapsack.x_s1  

                            # print ("x_s2", x_s2.shape)
                            # print ("x_s1", x_s1.shape)
                            # print ("c_batch", c_batch.shape)
                            # print ("penalty_batch", penalty_batch.shape)
                            # print ("purchase_fee", purchase_fee)
                            # print ("compensation_fee", compensation_fee)

                            loss = - (purchase_fee * (x_s2 * c_batch).sum() - (compensation_fee - purchase_fee) * torch.dot(c_batch, abs(x_s2-x_s1).float()))
                            batch_loss += loss
                        except:
                            print (f"Instance {batch} failed!")
                            print ("moving on")
                            continue
                self.log('train_loss', batch_loss, prog_bar=False, on_epoch=True, on_step=False)
                self.manual_backward(batch_loss)
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()


        for predictor in self.constr_predictors.values():
            predictor.eval()
        if self.predict_cost:
            self.cost_predictor.eval()

        