import time
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from OptProblems.knapsack.kpsolver import knapsack_solver

class knapsack_dataset(Dataset):
    """
    A dataset for knapsack problem

    Args:
        X (np.ndarray): feature matrix shape (num_data, num_features)
        w (np.ndarray): weights of knapsack items shape (num_data, dim, num_items)
        capacity (np.ndarray): capacity of knapsack shape (dim)
        costs (np.ndarray): value of knapsack problem shape (num_data, num_items)
        num_items (int): number of items
    """
    def __init__(self, X, w,  capacity, costs, num_items):
        self.X = X
        self.w = w
        self.costs = costs
        self.capacity = capacity
        self.solver = knapsack_solver(num_items)
        self.sols, self.objs = self._getSols()
        self.penaltyVector = np.ones_like(self.costs)
    
    def _getSols(self):
        sols = []
        objs = []
        for i in tqdm(range(len(self.X))):
            sol = self.solver.solve((self.w[i], self.capacity[i]), self.costs[i])
            sols.append(sol)
            objs.append([np.dot(self.costs[i], sol)])
        #     print (sol)
        # print (objs)
        return np.array(sols), np.array(objs)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        """
        A method to retrieve data

        Args:
            index (int): data index

        Returns:    
            tuple: data features (torch.tensor), (weights, capacity) as tuple of torch.tensor, 
            cost (torch.tensor), optimal solutions (torch.tensor) and objective values (torch.tensor)
        """
        return (
            torch.FloatTensor(self.X[index]),
            (torch.FloatTensor(self.w[index]), torch.FloatTensor(self.capacity[ index])),
            torch.FloatTensor(self.costs[index]),
            torch.FloatTensor(self.sols[index]),
            torch.FloatTensor(self.objs[index]),
            torch.FloatTensor(self.penaltyVector[index]),
        )