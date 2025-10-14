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

class alloy_solver (opt.optSolver):
    def __init__(self, n_items = 10, modelSense = opt.MINIMIZE):
        super().__init__( modelSense)
        self.n_items = n_items
        self.modelSense = modelSense

    def solve(self,  param_constraints, param_objective):
        """
        Args:
            param_constraints (tuple of np.ndarray): 
                weights (np.ndarray): shape (dim, num_items): weights of knapsack items
                req (np.ndarray) shape (dim): capacity of knapsack
            param_objective (np.ndarray) shape (num_items): value of knapsack problem
        Returns:
            x (np.ndarray): solution of knapsack problem
        Optimization Problem:
        Minimize:        cost^T x
        Subject to:      weights^T x ≥ req
                         x ≥ 0
        Variables:
        - x ∈ ℝ^n : decision variables
        - cost ∈ ℝ^n : cost vector
        - weights ∈ ℝ^n : estimated metal concentrations
        - req ∈ ℝ :  must acquire at least req amount of each metal
        
        Objective:
        Find x* = argmin_x cost.T @ x
        such that weights.T @ x >= req and x >= 0

        """

        weights, req = param_constraints
        # print("weights: ", weights , "req: ", req)
        costs = param_objective
        m = gp.Model("knapsack")
        x = m.addVars(self.n_items, name="x", vtype=GRB.INTEGER)
        m.setParam("OutputFlag", 0)
        for dim in range (len(req)):
            m.addConstr(gp.quicksum(weights[dim,j] * x[j]
                        for j in range(self.n_items)) >= req[dim])
        m.setObjective( gp.quicksum(costs[i] * x[i] for i in range(self.n_items)), GRB.MINIMIZE)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            if isinstance(x, gp.MVar):
                sol = x.X
            else:
                sol = [x[k].x for k in x]
            # print (sol)
            return np.array(sol)
        else:
            print(f"Model is not optimal, status code: {m.Status}, status: {status_dict.get(m.Status, 'UNKNOWN')}")
            # print("cost: ", costs   )
            # print("weights: ", weights   )
            # print("capacity: ", req   )
            raise Exception("No soluton found")

    def check_feasibility(self, param_constraints, sol):
        weights, req = param_constraints
        for dim in range(len(req)):
            if np.dot(weights[dim], sol) < req[dim]:
                return False
        return True

    def evaluate_solution(self, param_objective, param_constraints, sol):
        return np.dot(param_objective, sol)
    
    def correct_feasibility(self, param_constraints, sol):
        '''
        If req [dim] > np.dot(weights[dim], sol), then we need to scale up the solution
        '''
        weights, req = param_constraints
        max_tau = 1.0
        for dim in range(len(req)):
            tau = req[dim]/ np.dot(weights[dim], sol)
            max_tau = max(max_tau, tau)
        
        return sol * max_tau
    
    def denormalize (self, predicted_param_constraints, true_param_constraints):
        """
        Predicted weight is a list, The first item is the normalized weights
        We will rescale by multilying the true capacity and then send back as a list
        """

        normalized_weights , req= predicted_param_constraints 
        weights, req = true_param_constraints

        
        denormalized_weights = normalized_weights * einops.rearrange(req, 'batch dim -> batch dim 1')        
        return [denormalized_weights, req]

    def violation(self, param_constraints,  sol, all_comparisons = False):
        """
        Args:
            param_constraints (tuple of tensor): 
                weights (tensor): shape (batch_size, dim, num_items): weights of knapsack items
                req (tensor): shape (batch_size, dim): capacity of knapsack
            sol (tensor): shape (N, num_items): N candidate solution of knapsack problem
        Returns:
            excess_capacity (tensor): excess capacity for each dimension of shape (batch_size * N, dim) if all_comparisons is True else (batch_size, dim)
        """
        weights, req = param_constraints
        # Calculate constraint violation for each batch and dimension

        # Expand dimensions for broadcasting
        if all_comparisons:
            first_exp = einops.rearrange(weights, "batch dim num_items -> batch 1 dim num_items")  # Shape: [batch_size, 1, 4, 3]
            second_exp = einops.rearrange(sol, "n num_items -> 1 n 1 num_items")  # Shape: [1, N, 1, 3]
            # Compute dot product along the last dimension
            out = torch.einsum("bnic, bnic -> bni", first_exp, second_exp) 

            # Subtract capacity
            deficiency =  einops.rearrange(req, "batch dim -> batch 1 dim") - out # Shape: [batch_size, N, 4]

            # Reshape to (batch_size * N, 4)
            deficiency = einops.rearrange(deficiency, "batch n dim -> (batch n) dim")
            return deficiency
        else:
            # If weights.shape = [2, 3, 4] (batch=2, dim=3, items=4)
            # and sol.shape = [2, 4] (batch=2, items=4)
            # and req.shape = [2, 3] (batch=2, dim=3)
            # Then:
            # 1. weights * sol[:, None] creates [2, 3, 4] tensor where each item weight is multiplied by solution
            # 2. einops.reduce(..., "b d i -> b d", "sum") sums over items (i) dimension, resulting in [2, 3]
            # 3. Subtract req [2, 3] to get constraint violations for each dimension
            deficiency = req - einops.reduce(weights * sol[:, None], "b d i -> b d", "sum")
            return deficiency

    def constraint_wise_feasibility(self, param_constraints, sol, all_comparisons=False):
        out  = self.violation (param_constraints, sol, all_comparisons=all_comparisons)
        return torch.where(out <= 0, torch.zeros_like(out), torch.ones_like(out))

    def check_optimality(self, param_objective, obj, sol, all_comparisons=False, normalize = True):
        '''
        The purpose is to check if the candidate solution is suboptimal, it should be suboptimal.
        If it is return 0, otherwise return the margin by which it exceeds the optimal solution
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
            out = repeated_obj - dot_product
            if normalize:
                out = out / torch.abs(repeated_obj)
            out = einops.rearrange(out, "b n -> (b n)")
            return out
        else:
            # Compute dot product between (b,dim) and (b,dim) to get a shape (b,)   
            dot_product = einops.reduce(param_objective * sol, "b d -> b", "sum")
            out =  obj - dot_product.flatten()
            if normalize:
                out = out / torch.abs(obj) 
            # out = torch.where(dot_product <= obj, torch.zeros_like(dot_product), dot_product - obj)
            return out
        