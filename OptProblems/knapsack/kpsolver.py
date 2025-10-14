import numpy as np
import gurobipy as gp
from gurobipy import GRB
from OptProblems import opt
import einops
import torch
import torch.nn as nn
status_dict = {
            2: "OPTIMAL",
            3: "INFEASIBLE",
            4: "INF_OR_UNBD",
            5: "UNBOUNDED",
            6: "CUTOFF",
            7: "ITERATION_LIMIT",
            8: "NODE_LIMIT",
            9: "TIME_LIMIT",
            10: "SOLUTION_LIMIT",
            11: "INTERRUPTED",
            12: "NUMERIC",
            13: "SUBOPTIMAL"
        }
class knapsack_solver (opt.optSolver):
    def __init__(self, n_items, modelSense = opt.MAXIMIZE):
        super().__init__( modelSense)
        self.n_items = n_items
        self.modelSense = modelSense

    
    def solve(self,  param_constraints, param_objective):
        """
        Args:
            param_constraints (tuple of np.ndarray): 
                weights (np.ndarray): shape (dim, num_items): weights of knapsack items
                capacity (np.ndarray) shape (dim): capacity of knapsack
            param_objective (np.ndarray) shape (num_items): value of knapsack problem
        Returns:
            x (np.ndarray): solution of knapsack problem
        """

        weights, capacity = param_constraints
        costs = param_objective
        # print ("weights: ", weights)
        # print ("capacity: ", capacity)
        # print ("Solving knapsack problem...")

        m = gp.Model("knapsack")
        x = m.addVars(self.n_items, name="x", vtype=GRB.BINARY)
        m.setParam("OutputFlag", 0)
        m.setParam('TimeLimit', 10)
        for dim in range (len(capacity)):
            m.addConstr(gp.quicksum(weights[dim,j] * x[j]
                        for j in range(self.n_items)) <= capacity[dim])
        m.setObjective( gp.quicksum(costs[i] * x[i] for i in range(self.n_items)), GRB.MAXIMIZE)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            if isinstance(x, gp.MVar):
                return x.X
            else:
                sol = [x[k].x for k in x]
                return np.array(sol)
        else:
            # print ("Status of Optimization: ", m.Status)
            # print(f"Model is not optimal, status code: {m.Status}, status: {status_dict.get(m.Status, 'UNKNOWN')}")
            # print("cost: ", costs   )
            # print("weights: ", weights   )
            # print("capacity: ", capacity   )
            raise Exception("No soluton found")

    def check_feasibility(self, param_constraints, sol):
        weights, capacity = param_constraints
        for dim in range(len(capacity)):
            # print ("Inside feasibility check", np.dot(weights[dim], sol), capacity[dim])
            if np.dot(weights[dim], sol) > capacity[dim]:
                return False
        return True

    def evaluate_solution(self, param_objective,param_constraints,  sol):
        return np.dot(param_objective, sol)
    
    def correct_feasibility(self, param_constraints, sol):
        weights, capacity = param_constraints
        max_tau = 1.0
        for dim in range(len(capacity)):
            tau =  np.dot(weights[dim], sol)/ capacity[dim]
            max_tau = max(max_tau, tau)
        
        return sol / max_tau
    
    def denormalize (self, predicted_param_constraints, true_param_constraints):
        """
        Predicted weight is a list, The first item is the normalized weights
        We will rescale by multilying the true capacity and then send back as a list
        """

        normalized_weights , capacity= predicted_param_constraints 
        weights, capacity = true_param_constraints

        
        denormalized_weights = normalized_weights * einops.rearrange(capacity, 'batch dim -> batch dim 1')        
        return [denormalized_weights, capacity]

        
    
    def violation(self, param_constraints,  sol, all_comparisons = False):
        """
        Args:
            param_constraints (tuple of tensor): 
                weights (tensor): shape (batch_size, dim, num_items): weights of knapsack items
                capacity (tensor): shape (batch_size, dim): capacity of knapsack
            sol (tensor): shape (N, num_items): N candidate solution of knapsack problem
        Returns:
            excess_capacity (tensor): excess capacity for each dimension of shape (batch_size * N, dim) if all_comparisons is True else (batch_size, dim)
        """
        weights, capacity = param_constraints
        # Calculate constraint violation for each batch and dimension

        # Expand dimensions for broadcasting
        if all_comparisons:
            first_exp = einops.rearrange(weights, "batch dim num_items -> batch 1 dim num_items")  # Shape: [batch_size, 1, 4, 3]
            second_exp = einops.rearrange(sol, "n num_items -> 1 n 1 num_items")  # Shape: [1, N, 1, 3]
            # Compute dot product along the last dimension
            out = torch.einsum("bnic, bnic -> bni", first_exp, second_exp) 

            # Subtract capacity
            out = out - einops.rearrange(capacity, "batch dim -> batch 1 dim")  # Shape: [batch_size, N, 4]

            # Reshape to (batch_size * N, 4)
            out = einops.rearrange(out, "batch n dim -> (batch n) dim")
            return out
        else:
            # If weights.shape = [2, 3, 4] (batch=2, dim=3, items=4)
            # and sol.shape = [2, 4] (batch=2, items=4)
            # and capacity.shape = [2, 3] (batch=2, dim=3)
            # Then:
            # 1. weights * sol[:, None] creates [2, 3, 4] tensor where each item weight is multiplied by solution
            # 2. einops.reduce(..., "b d i -> b d", "sum") sums over items (i) dimension, resulting in [2, 3]
            # 3. Subtract capacity [2, 3] to get constraint violations for each dimension
            excess_capacity = einops.reduce(weights * sol[:, None], "b d i -> b d", "sum") - capacity
            return excess_capacity

    def constraint_wise_feasibility(self, param_constraints, sol, all_comparisons=False):
        '''
        return zero, if no violation of constaints else 1
        '''
        out  = self.violation (param_constraints, sol, all_comparisons=all_comparisons)
        return torch.where(out <= 0, torch.zeros_like(out), torch.ones_like(out))

    def check_optimality(self, param_objective, obj, sol, all_comparisons=False, normalize = False):
        '''
        The purpose is to check if the candidate solution is suboptimal, it should be suboptimal.
        This function returns suboptimalty, (should be negative, but may be postive for infeasible solutions)
        Args:
            param_objective (tensor): shape (batch_size, num_items): value of knapsack problem
            obj (tensor): shape (batch_size,): objective value of knapsack problem
            sol (tensor): shape ( N, num_items): N candidate solutions of knapsack problem
            all_comparisons (bool): whether to check all comparisons
        Returns:
            torch.tensor: returns 0 if objective value with candidate solution is suboptimal else 1
            returns tensor of shape (batch_size * N, dim) if all_comparisons is True else (batch_size, dim)
        '''
        if all_comparisons:
            dot_product = einops.einsum(
                param_objective, sol, "b d, n d -> b n"
            )  # Shape: (b, N)

            repeated_obj = einops.repeat(obj, "b 1 -> b n", n=sol.shape[0])
            # out = torch.where(dot_product <= repeated_obj, torch.zeros_like(dot_product), 
            #     dot_product - repeated_obj)
            out = dot_product - repeated_obj
            if normalize:
                out = out / torch.abs(repeated_obj)
            out = einops.rearrange(out, "b n -> (b n)")
            return out
        else:
            # Compute dot product between (b,dim) and (b,dim) to get a shape (b,)   
            dot_product = einops.reduce(param_objective * sol, "b d -> b", "sum")
            out =  dot_product - obj.flatten()

            if normalize:
                out = out / torch.abs(obj) 
            # out = torch.where(dot_product <= obj, torch.zeros_like(dot_product), dot_product - obj)
            return out
        


    # def loss_positive_sample(self, param_constraints, param_objective, pos_sol):
    #     return nn.GELU()(self._constriantwisevalue_torch(param_constraints, param_objective, pos_sol) )
    
    # def loss_negative_sample(self, param_constraints, param_objective, neg_sol):
    # def loss_negative_sample(self, param_constraints, param_objective, neg_sol):
    # def loss_negative_sample(self, param_constraints, param_objective, neg_sol):
    #     return  -nn.GELU()  ( self._constriantwisevalue_torch(param_constraints, param_objective, neg_sol) )
