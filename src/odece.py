import numpy as np
import torch
from torch.autograd import Function
from torch import nn
from datetime import datetime
from OptProblems import opt
import pytorch_lightning as pl
import traceback
from src.pfl import PFL
from src.odece_utils import dirac_GaussianApprox, ParetoOptimalAlpha, weightedAlpha, PCGrad

class ODECE(PFL): 
    """
    A PyTorch Lightning module for Optimizing Decision through End-to-End Constraint Estimation (ODECE).

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
        processes (int): Number of processors
        dataset (Dataset): Training data if caching is used
        
        infeasibility_aversion_coeff (float): Coefficient for infeasibility aversion. Default is 0.5
        
        lr (float): Learning rate for optimizers
        max_epochs (int): Maximum number of training epochs
        scheduler: Learning rate scheduler
        seed (int): Random seed
    """
    def __init__(self, ml_predictor, optsolver, num_predconstrsvar,
        predict_indices=None, predict_cost=False, 
        margin_threshold= 2., denormalize = False,
        infeasibility_aversion_coeff =0.5, save_instance_wise_metrics=True,
        processes=1,   dataset=None,
        lr=1e-3, max_epochs=100, scheduler=None, seed=135):
        super().__init__(ml_predictor, optsolver, num_predconstrsvar, predict_indices, predict_cost,
             denormalize, save_instance_wise_metrics, processes,  dataset, lr, max_epochs, scheduler, seed)
        # self.automatic_optimization = True

        # self.dflloss = dfl_method
        self.margin_threshold = margin_threshold
        self.prev_fpl_loss_value = 0.
        
        self.num_warmup_epochs = 0
        
        self.save_hyperparameters('num_predconstrsvar', 'predict_indices', 'predict_cost', 
        'denormalize', 'max_epochs', 'lr', 'scheduler', 'seed',     
         'margin_threshold', 'infeasibility_aversion_coeff')

    def losses_computation(self, trueConstr_tuple, pred_params_tuple, costs, sols, objs, pred_sols, mask):
        

        violation_true = self.optsolver.constraint_wise_feasibility(trueConstr_tuple, sols)
        ######################## SANITY CHECK ########################
        # print ("violation_true", violation_true) ### All should be 0
        ######################## SANITY CHECK ########################
        loss_wrt_true = self.optsolver.violation(pred_params_tuple, sols )

        loss_withtruesol = (
            (1 - violation_true) * (nn.Softplus(beta=5)(loss_wrt_true + self.margin_threshold)) # Feasible Solutions, reduce excess capacity
        )

        loss_withtruesol = (torch.mean (loss_withtruesol, dim =1)) # FPL mean
        # print ("Check shape of FPL", loss_withtruesol.shape)  
        # print ("Loss FPL", loss_withtruesol)

        self.log('loss_fpl', loss_withtruesol.mean(), prog_bar=False, on_epoch=True, on_step=False) 

        # print ("Is infeasible: ", violation_pred)
        if mask.sum() == 0:
            print ("No feasible solutions found")
            # Create zero losses that maintain gradient flow
            device = mask.device
            batch_size = mask.shape[0]
            # Get a parameter tensor to maintain gradient flow
            param_sum = sum(p.sum() for p in pred_params_tuple)
            loss_ial = torch.zeros_like(param_sum).expand(batch_size) * param_sum * 0
            return (
                   loss_withtruesol,  # Keep original loss_withtruesol
                   loss_ial
                   )
        masked_trueConstr_tuple = tuple(val[mask] for val in trueConstr_tuple)
        masked_pred_params_tuple = tuple(val[mask] for val in pred_params_tuple)
        
        
        violation_pred = self.optsolver.constraint_wise_feasibility(masked_trueConstr_tuple, 
            pred_sols, all_comparisons=False)
        ######################## SANITY CHECK ########################
        # print ("violation_pred", violation_pred)
        ######################## SANITY CHECK ########################

        
        # print (violation_pred.shape)
        ### Zero if no violation else 1
        # print ("violation_pred", violation_pred.shape)
        violation_pred_mask = (violation_pred.sum(dim=1) >=1 )
        ### violation_pred_mask is True if at least one constraint is violated

        

        loss_wrt_pred = self.optsolver.violation(masked_pred_params_tuple, pred_sols, 
            all_comparisons=False)
        ######################## SANITY CHECK ########################
        # print ("loss_wrt_pred", loss_wrt_pred)
        ######################## SANITY CHECK ########################

        # print (loss_wrt_pred.shape)

        loss_ial = violation_pred * nn.Softplus(beta=5)(    (-loss_wrt_pred + self.margin_threshold) )
        # loss_ial = dirac_GaussianApprox(  loss_wrt_pred )
        loss_ial = loss_ial.sum(dim=1) / (violation_pred.sum(dim=1).clamp(min=1)) # IAL

        

        
        optimalityviolation_pred = self.optsolver.check_optimality(costs [mask], objs [mask], 
            pred_sols, all_comparisons= False, normalize = False)
        indices = (optimalityviolation_pred >= 0).nonzero(as_tuple=True)[0]

        
        loss_wol = torch.ones_like(optimalityviolation_pred)
        loss_ipl = ( loss_wol * loss_ial )
        ## Only consider negative solutions,  i.e, at least one constraint is violated
        ### If no constraint is violated it is not a negative solution
        loss_ipl = loss_ipl [violation_pred_mask] #[indices]
        self.log('loss_ipl', loss_ipl.mean(), prog_bar=False, on_epoch=True, on_step=False)

        
        return  loss_ipl , loss_withtruesol


    
    def _batchsolve (self, pred_costs, pred_params_tuple, trueConstr_tuple, sol_len, batch_size):
        predicted_cost_params_np = pred_costs.detach().cpu().numpy()

        predicted_constrs_params_np = [param.detach().cpu().numpy() for param in pred_params_tuple]
        true_constrs_params_np = [param.detach().cpu().numpy() for param in trueConstr_tuple]
        # Create a list to store all predicted solutions
        # all_pred_sols = []
        pred_sols_tensor = torch.empty((batch_size, sol_len), device=pred_costs.device)
        mask = torch.zeros(batch_size, dtype=torch.bool, device=pred_costs.device)

        for b in range(batch_size):
            pred_params =  tuple(param[b] for param in predicted_constrs_params_np)
            true_params = tuple(param[b] for param in true_constrs_params_np)
            try:
                pred_sol = self.optsolver.solve( pred_params, predicted_cost_params_np[b])
                # print (pred_sol)
                # all_pred_sols.append(pred_sol)
                pred_sols_tensor[b] = torch.from_numpy(pred_sol).float()
                mask[b] = True
            except:
                # No solution found, may be infeasible
                mask[b] = False
                # print ("ERROR")
        
        return pred_sols_tensor[mask], mask

    def training_step(self, batch, batch_idx):
        """ 
        Args:
            batch: Tuple of (features, trueConstr_tuple, costs, sols, objs) where:
                - features: Input features
                - trueConstr_tuple: Tuple of constraint parameters
                - costs: Cost parameters (used if not predicting costs)
                - sols: True solutions
                - objs: True objective values
                - penalty: Penalty vector values (only used for Post-hoc regret computation)
            batch_idx: Index of current batch

        """

        # Get optimizer and ensure it's a list
        
        # Get optimizer and ensure it's a list
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        features, trueConstr_tuple, costs, sols, objs, penalty = batch
        batch_size = len(features)

        with torch.no_grad():
            predicted_params = self.forward(features)
            pred_params_tuple, pred_costs = self._create_params_tuple(predicted_params,
                                                                    trueConstr_tuple, costs)
            predsol, mask = self._batchsolve (pred_costs, 
                                                    pred_params_tuple, 
                                                    trueConstr_tuple, 
                                                    sols.shape[1], 
                                                    batch_size)
        for predictor in self.constr_predictors.values():
            predictor.train()
        if self.predict_cost:
            self.cost_predictor.train()
        
        for i, predictor in self.constr_predictors.items():
            optimizer = optimizers[self.idx_to_opt_pos[i]]
            optimizer.zero_grad()
            model = self.constr_predictors[i]  
            predicted_params = self.forward(features)
            pred_params_tuple, pred_costs = self._create_params_tuple(predicted_params,
                                                                    trueConstr_tuple, costs)

            loss_ipl, loss_fpl = self.losses_computation( 
                                                            trueConstr_tuple, 
                                                            pred_params_tuple, 
                                                            costs, sols, objs, 
                                                            predsol, mask)

            self._update_grad(loss_ipl, loss_fpl, model, optimizer)
        for predictor in self.constr_predictors.values():
            predictor.eval()
        if self.predict_cost:
            self.cost_predictor.eval()

    def _update_grad(self, loss_ipl, loss_fpl, model, optimizer):
        prev_fpl_loss_value = loss_fpl.mean()
        prev_ipl_loss_value = loss_ipl.mean()
        total_loss = (
            self.hparams.infeasibility_aversion_coeff *(prev_ipl_loss_value + 1e-4).mean()
            + (1 - self.hparams.infeasibility_aversion_coeff) *(prev_fpl_loss_value + 1e-4).mean()
        )
        self.log ('train_loss', total_loss, prog_bar=False, on_epoch=True, on_step=False)

        self.manual_backward(total_loss, retain_graph=True)

        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
    def on_train_epoch_end(self) :
        """Log the elapsed time at the end of each training epoch"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.log("elapsedtime", elapsed, prog_bar=False, on_step=False, on_epoch=True)
        