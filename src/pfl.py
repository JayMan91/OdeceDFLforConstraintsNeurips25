import numpy as np
import torch
from torch.autograd import Function
from torch import nn
from datetime import datetime
from OptProblems import opt
import pytorch_lightning as pl
import traceback
import os
import json

class PFL(pl.LightningModule):
    """
    Predictive First Layer for optimization problems.
    
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
        denormalize (bool): Whether to denormalize the predicted parameters.
        processes (int): Number of processorse
        dataset (Dataset): Training data if caching is used
        lr (float): Learning rate for optimizers
        max_epochs (int): Maximum number of training epochs
        scheduler: Learning rate scheduler
        seed (int): Random seed
    """
    def __init__(self, ml_predictor, optsolver, num_predconstrsvar, predict_indices=None, predict_cost=False,
        denormalize =False, save_instance_wise_metrics=True, processes=1,  dataset=None, 
        lr=1e-3, max_epochs=100, scheduler=None, seed=135):
        super().__init__()
        torch.manual_seed(seed)
        if predict_indices is None:
            predict_indices = sorted(range(num_predconstrsvar))  
        else:
            if isinstance(predict_indices, int):
                predict_indices = [predict_indices]

            predict_indices = sorted(predict_indices)  # Ensure consistent ordering
        
        # Convert ml_predictor list to dictionary mapping indices to their predictors
        if isinstance(ml_predictor, (list, tuple)):
            if predict_cost:
                assert len(ml_predictor) == len(predict_indices) + 1
                self.cost_predictor = ml_predictor[-1]
                ml_predictor = ml_predictor[:-1]
            else:
                assert len(ml_predictor) == len(predict_indices)
            
            # Map each index to its predictor and store in constr_predictors
            # This dictionary maps each index in predict_indices to its corresponding predictor model
            self.constr_predictors = {idx: pred for idx, pred in zip(predict_indices, ml_predictor)}
            # Create reverse mapping from actual index to position in predict_indices
            # This dictionary maps each index in predict_indices to its position in the list
            self.idx_to_opt_pos = {idx: pos for pos, idx in enumerate(predict_indices)}
        else:
            raise ValueError("ml_predictor must be a list or tuple of predictors for the given predict_indices")
        
        self.predict_indices = predict_indices
        self.num_predconstrsvar = num_predconstrsvar
        self.optsolver = optsolver
        self.predict_cost = predict_cost
        self.processes = processes
        self.denormalize = denormalize
        self.dataset = dataset
        self.save_instance_wise_metrics = save_instance_wise_metrics
        self.save_hyperparameters('num_predconstrsvar', 'predict_indices', 'predict_cost', 
        'denormalize', 'max_epochs', 'lr', 'scheduler', 'seed')
        self.automatic_optimization = False # update manually
        self.validation_step_outputs = []
        self.start_time = datetime.now()


    def _create_params_tuple(self, predicted_params, trueConstr_tuple, costs):
        # if self.denormalize:
        #     predicted_params = self.optsolver.denormalize (predicted_params, trueConstr_tuple)
        # print("Weights: ",predicted_params[0] [0])
        # Get cost parameter (either predicted or true)
        if self.predict_cost:
            pred_costs = predicted_params[-1]
        else:
            pred_costs = costs
        
        # Create full parameter tuple (predicted + true values)
        pred_constrs_list = []
        for i in range(len(trueConstr_tuple)):
            if i in self.predict_indices:
                # Use predicted parameter for this index
                pred_idx = self.idx_to_opt_pos[i]
                pred_constrs_list.append(predicted_params[pred_idx])
            else:
                # Use true parameter for this index
                pred_constrs_list.append(trueConstr_tuple[i])
        
        # Create tuple of predicted parameters (predicted for predict_indices, true for others)
        pred_params_tuple = tuple(pred_constrs_list)
        if self.denormalize:
            pred_constrs_list = self.optsolver.denormalize (pred_params_tuple, trueConstr_tuple)
            pred_params_tuple = tuple(pred_constrs_list)
        
        return pred_params_tuple, pred_costs

    def forward(self, features):
        """Generates predictions for all variables in predict_indices order.
        
        Uses constr_predictors to generate predictions for each variable index in sorted order.
        For example, if predict_indices=[2,0,5], returns predictions in order [0,2,5].
        If predict_cost is True, appends cost prediction at the end.
        
        Args:
            features: Input features for prediction
            
        Returns:
            list: Predicted parameters in sorted order of predict_indices.
                 If predict_cost is True, cost prediction is appended at the end.
        """
        # Get predictions for variables in sorted index order
        predicted_params = [self.constr_predictors[idx](features) for idx in sorted(self.constr_predictors.keys())]
        
        # Append cost prediction if enabled
        if self.predict_cost:
            predicted_params.append(self.cost_predictor(features))
            
        return predicted_params
    
    def training_step(self, batch, batch_idx):
        """ 
        Args:
            batch: Tuple of (features, trueConstr_tuple, costs, sols, objs) where:
                - features: Input features
                - trueConstr_tuple: Tuple of constraint parameters
                - costs: Cost parameters (used if not predicting costs)
                - sols: True solutions
                - objs: True objective values
                - penalty: True penalty values (shape same as costs)
            batch_idx: Index of current batch

        """
        features, trueConstr_tuple, costs, sols, objs, penalty = batch
        
        # Get optimizer and ensure it's a list
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        

        for predictor in self.constr_predictors.values():
            predictor.train()
        if self.predict_cost:
            self.cost_predictor.train()
            
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
            
        # Train cost predictor if enabled
        if self.predict_cost:
            ml_predictor = self.cost_predictor
            predicted_params = ml_predictor(features)
            
            # Cost optimizer is always last in the list
            optimizer = optimizers[-1]
            
            optimizer.zero_grad()
            loss = criterion(predicted_params, costs)
            self.manual_backward(loss)
            optimizer.step()
            self.log('train_loss_cost', loss, prog_bar=True, on_epoch=True, on_step=False)

        for predictor in self.constr_predictors.values():
            predictor.eval()
        if self.predict_cost:
            self.cost_predictor.eval()

        

    def on_train_epoch_end(self) :
        """Log the elapsed time at the end of each training epoch"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.log("elapsedtime", elapsed, prog_bar=False, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx, prefix="val"):
        """
        Args:
            batch: Input batch containing features and true parameters
            batch_idx: Index of the current batch
            prefix: Prefix for logging metrics ('val' or 'test')
        """
        features, trueConstr_tuple, costs, sols, objs, penalty = batch
        # for v in trueConstr_tuple:
        #     print ("shape", v.shape)
        # print ("Inputs Shape: ", features.shape, costs.shape, sols.shape, objs.shape)

        # print (self.idx_to_opt_pos)
        
        # set models to evaluation mode
        for predictor in self.constr_predictors.values():
            predictor.eval()
        if self.predict_cost:
            self.cost_predictor.eval()
        

        predicted_params = self.forward(features)
        criterion = nn.MSELoss()
        if self.predict_cost:
            loss = criterion( predicted_params[-1], costs)
            self.log(f'{prefix}_loss_cost', loss, prog_bar=False)

        # Create full parameter tuple (predicted + true values)
        
        pred_constrs_list = []
        for i in range(len(trueConstr_tuple)):
            if i in self.predict_indices:
                # Use predicted parameter for this index
                pred_idx = self.idx_to_opt_pos[i]
                pred_constrs_list.append(predicted_params[pred_idx])
            else:
                # Use true parameter for this index
                pred_constrs_list.append(trueConstr_tuple[i])
        
        # Create tuple of predicted parameters (predicted for predict_indices, true for others)
        pred_params_tuple = tuple(pred_constrs_list)
        
        if self.denormalize:
            pred_constrs_list = self.optsolver.denormalize (pred_params_tuple, trueConstr_tuple)
            pred_params_tuple = tuple(pred_constrs_list)

        
        # Compute losses for each predicted parameter
        for idx in sorted(self.constr_predictors.keys()):
            loss = criterion(pred_params_tuple[idx], trueConstr_tuple[idx])
            self.log(f'{prefix}_loss_constraint_{idx}', loss, prog_bar=False)
        
        ### Regret, Post-hoc Regret  and Infeasibility Computations
        ### NOTE: There mightbe more number of variables than the number of predicted variables
        ### In that case, we are not predicting all the variables
        ### For the remaining variables, we use the true values of the variables to compute the solution
        
        batch_Preregret = 0
        batch_posthoc_regret = 0
        batch_infeasibility = 0
        
        # Pre-process all samples at once for efficiency
        batch_size = len(features)
        
        # Convert predictions to numpy arrays in order of predict_indices
        # If predict_cost is True, exclude the last prediction (cost prediction)
        # predicted_constrs_params_np = [param.detach().cpu().numpy() for param in predicted_params[:-1 if self.predict_cost else None]]
        
        if self.predict_cost:
            # Get cost prediction (last element of predicted_params)
            predicted_cost_params_np = predicted_params[-1].detach().cpu().numpy()
        
        # Convert true parameters to numpy arrays
        true_constrs_params_np = [param.detach().cpu().numpy() for param in trueConstr_tuple]
        pred_constrs_params_np = [param.detach().cpu().numpy() for param in pred_params_tuple]
        true_costs_np = costs.detach().cpu().numpy()
        true_sols_np = sols.detach().cpu().numpy()
        true_objs_np = objs.detach().cpu().numpy()
        penalty_np = penalty.detach().cpu().numpy()
        
        regrets = []
        posthoc_regrets = []
        infeasibilities = []
        unsolvables = []
        recourse_costs = []
        infeasible_regrets = []
        is_true_feasible = []

        
        for l in range(batch_size):
            pred_params = tuple(param[l] for param in pred_constrs_params_np)
            true_params = tuple(param[l] for param in true_constrs_params_np)
            true_obj =  true_objs_np[l][0]
            penalty_coeff = penalty_np[l].mean()  

            true_feasibility = self.optsolver.check_feasibility( pred_params, true_sols_np[l])
            if true_feasibility:
                is_true_feasible.append(1)
            else:
                is_true_feasible.append(0)
            try:

                if self.predict_cost:
                    pred_sol = self.optsolver.solve( pred_params, predicted_cost_params_np[l])
                else:
                    pred_sol = self.optsolver.solve( pred_params, true_costs_np[l])

                # Evaluate the realized objective value with predicted solution
                realized_obj_w_pred = self.optsolver.evaluate_solution(true_costs_np[l], true_params, pred_sol)
                # print ("Realized Objective", realized_obj_w_pred)
                # print ("True Objective", true_obj)
                # print ("Predicted Parameters", pred_params)
                # print ("True Parameters", true_params)
                # print ("Predicted Solution", pred_sol)
                # print ("True Solution", true_sols_np[l])
                # Calculate regret as the difference between the true objective and the realized objective
                regret =  (realized_obj_w_pred - true_obj) / true_obj
                if self.optsolver.modelSense == opt.MAXIMIZE:
                    regret = -regret
                # print ("Regret", regret, true_obj, realized_obj_w_pred  )
                
                
                # Check feasibility of the predicted solution with respect to the true parameters
                feasibility = self.optsolver.check_feasibility( true_params, pred_sol)

                if not feasibility:
                    # If the solution is infeasible, correct it and calculate the post-hoc regret
                    infeasibilities.append(1)
                    corrected_sol = self.optsolver.correct_feasibility( true_params, pred_sol)
                    corrected_obj_w_pred = self.optsolver.evaluate_solution(true_costs_np[l], true_params, corrected_sol)
                    corrected_regret =  (corrected_obj_w_pred - true_obj) / true_obj
                    if self.optsolver.modelSense == opt.MAXIMIZE:
                        corrected_regret = -corrected_regret
                    # Calculate penalty based on the difference between corrected and predicted solutions
                    
                    penalty_value=  (realized_obj_w_pred - corrected_obj_w_pred) / true_obj
                    penalty_value = np.abs(penalty_value)

                    posthoc_regrets.append(corrected_regret + penalty_coeff * penalty_value)
                    recourse_costs.append(penalty_value)
                    regrets.append(np.nan)
                    infeasible_regrets.append(regret)
                else:
                    # If feasible, add the regular regret
                    posthoc_regrets.append(regret)
                    infeasibilities.append(0)
                    regrets.append(regret)
                    infeasible_regrets.append(np.nan)
                    recourse_costs.append(0)
                unsolvables.append(0)

                
            except Exception as e:
                print("Error in solving optimization problem:", e)
                print(traceback.format_exc())  # Print the full traceback
                
                # print ("Predicted Parameters", pred_params)
                # print ("True Parameters", true_params)
                infeasibilities.append(np.nan)
                regrets.append(np.nan)
                recourse_costs.append(np.nan)
                infeasible_regrets.append(np.nan)
                unsolvables.append(1)
                posthoc_regrets.append( true_obj*penalty_coeff)

        regrets_np = np.array(regrets)
        batch_Preregret = np.nanmean(regrets_np)

        recourse_costs_np = np.array(recourse_costs)
        batch_recourse_cost = np.nanmean(recourse_costs)

        batch_posthoc_regret = np.nanmean(posthoc_regrets)
        infeasibilities_np = np.array(infeasibilities)
        batch_infeasibility = np.nanmean(infeasibilities_np)
        batch_unsolvable = np.nanmean(unsolvables)
        batch_infeasible_regret = np.nanmean(infeasible_regrets)
        batch_is_true_feasible = np.nanmean(is_true_feasible)

        batch_loss = {
            f'{prefix}_regret': batch_Preregret,
            f'{prefix}_posthoc_regret': batch_posthoc_regret,
            f'{prefix}_infeasibility': batch_infeasibility,
            f'{prefix}_unsolvable': batch_unsolvable,
            f'{prefix}_recourse_cost': batch_recourse_cost,
            f'{prefix}_infeasible_regret': batch_infeasible_regret,
            f'{prefix}_is_true_feasible': batch_is_true_feasible
        }
        if self.save_instance_wise_metrics and (prefix == "test"):
            start_idx = len(self.validation_step_outputs)
            indices = list(range(start_idx, start_idx + batch_size ))
            batch_loss["instance_indices"] = indices
            batch_loss["instance_regret"] = regrets_np
            batch_loss["instance_infeasibility"] = infeasibilities_np

        self.validation_step_outputs.append(batch_loss)
        return batch_loss

    def test_step(self, batch, batch_idx):
        """Runs validation step with test prefix."""
        return self.validation_step(batch, batch_idx, prefix="test")

    def _aggregate_and_log_epoch_metrics(self, prefix="val"):
        regrets = [x[f'{prefix}_regret'] for x in self.validation_step_outputs if x is not None and not np.isnan(x[f'{prefix}_regret'])]
        posthoc_regrets = [x[f'{prefix}_posthoc_regret'] for x in self.validation_step_outputs if x is not None and not np.isnan(x[f'{prefix}_posthoc_regret'])]
        infeasibilities = [x[f'{prefix}_infeasibility'] for x in self.validation_step_outputs if x is not None and not np.isnan(x[f'{prefix}_infeasibility'])]
        unsolvables = [x[f'{prefix}_unsolvable'] for x in self.validation_step_outputs if x is not None and not np.isnan(x[f'{prefix}_unsolvable'])]
        recourse_costs = [x[f'{prefix}_recourse_cost'] for x in self.validation_step_outputs if x is not None and not np.isnan(x[f'{prefix}_recourse_cost'])]
        infeasible_regrets = [x[f'{prefix}_infeasible_regret'] for x in self.validation_step_outputs if x is not None and not np.isnan(x[f'{prefix}_infeasible_regret'])]
        is_true_feasible = [x[f'{prefix}_is_true_feasible'] for x in self.validation_step_outputs if x is not None and not np.isnan(x[f'{prefix}_is_true_feasible'])]

        if len(regrets) > 0:
            self.log(f'{prefix}_regret', np.nanmean(regrets), prog_bar=True)
        if len(posthoc_regrets) > 0:
            self.log(f'{prefix}_posthoc_regret', np.nanmean(posthoc_regrets))
        if len(infeasibilities) > 0:
            self.log(f'{prefix}_infeasibility', np.nanmean(infeasibilities), prog_bar=True)
        if len(unsolvables) > 0:
            self.log(f'{prefix}_unsolvable', np.nanmean(unsolvables))
        if len(recourse_costs) > 0:
            self.log(f'{prefix}_recourse_cost', np.nanmean(recourse_costs))
        if len(infeasible_regrets) > 0:
            self.log(f'{prefix}_infeasible_regret', np.nanmean(infeasible_regrets))
        if len(is_true_feasible) > 0:
            self.log(f'{prefix}_is_true_feasible', np.nanmean(is_true_feasible))
        
        if self.save_instance_wise_metrics and (prefix == "test"):
            indices = []
            instance_regret = []
            instance_infeasibility = []
            for x in self.validation_step_outputs:
                if "instance_indices" in x:
                    indices.extend(x["instance_indices"])
                if "instance_regret" in x:
                    instance_regret.extend(x["instance_regret"].tolist())
                if "instance_infeasibility" in x:
                    instance_infeasibility.extend(x["instance_infeasibility"].tolist())

            test_instance_losses = {
                "instance_indices": indices,
                "instance_regret": instance_regret,
                "instance_infeasibility": instance_infeasibility
            }
            # print (test_instance_losses)
            # Save to the logger's log directory
            log_dir = self.trainer.logger.log_dir  # This is where hparams.yaml, etc. are stored
            save_path = os.path.join(log_dir, "instance_losses.json")
            with open(save_path, "w") as f:
                json.dump(test_instance_losses, f)
        
        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        self._aggregate_and_log_epoch_metrics(prefix="val")

    def on_test_epoch_end(self):
        self._aggregate_and_log_epoch_metrics(prefix="test")

    def configure_optimizers(self):
        """
        Returns:
            list: List of optimizers in predict_indices order, with cost optimizer last if predict_cost is True.
                 Always returns a list, even when there's only one optimizer.
        """
        optimizers = []
        # Create optimizers in the same order as predict_indices
        for idx in self.predict_indices:
            optimizer = torch.optim.Adam(self.constr_predictors[idx].parameters(), lr=self.hparams.lr)
            optimizers.append(optimizer)
        
        if self.predict_cost:
            optimizer = torch.optim.Adam(self.cost_predictor.parameters(), lr=self.hparams.lr)
            optimizers.append(optimizer)

        return optimizers