import time
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from OptProblems.alloyproduction.alloysolver import alloy_solver

req_values = [627.54, 369.72]
class alloy_dataset(Dataset):
    """
    A dataset for alloy production problem

    Returns:
        X (np.ndarray): feature matrix shape (num_data,dim, num_items, num_features = 4096)
        y (np.ndarray): target matrix shape ((num_data, dim, num_items)
        cost (np.ndarray): cost matrix shape ( num_data, num_items  )
        req (np.ndarray): req of knapsack shape (num_datadim)
        penalty (np.ndarray): penalty matrix shape (num_data, num_items)
    """
    def __init__(self, mode = 'train', rowSizeG = 2, colSizeG = 10, penaltyTerm = 0.25, seed = 135):
        testi = 0
        
        if mode == 'train':

            c = np.loadtxt('./data/Alloy production/brass/train_prices/train_prices(' + str(testi) + ').txt')
            x = np.loadtxt('./data/Alloy production/brass/train_features/train_features(' + str(testi) + ').txt')
            y = np.loadtxt('./data/Alloy production/brass/train_weights/train_weights(' + str(testi) + ').txt')
            penalty = np.loadtxt('./data/Alloy production/brass/train_penalty' + str(penaltyTerm) + '/train_penalty(' + str(testi) + ').txt')
        elif mode == 'test':
            c = np.loadtxt('./data/Alloy production/brass/test_prices/test_prices(' + str(testi) + ')_n.txt')
            x = np.loadtxt('./data/Alloy production/brass/test_features/test_features(' + str(testi) + ')_n.txt')
            y = np.loadtxt('./data/Alloy production/brass/test_weights/test_weights(' + str(testi) + ')_n.txt')
            penalty = np.loadtxt('./data/Alloy production/brass/test_penalty' + str(penaltyTerm) + '/test_penalty(' + str(testi) + ')_n.txt')

        elif mode == 'val':
            c = np.loadtxt('./data/Alloy production/brass/val_prices/val_prices(' + str(testi) + ').txt')
            x = np.loadtxt('./data/Alloy production/brass/val_features/val_features(' + str(testi) + ').txt')
            y = np.loadtxt('./data/Alloy production/brass/val_weights/val_weights(' + str(testi) + ').txt')
            penalty = np.loadtxt('./data/Alloy production/brass/val_penalty' + str(penaltyTerm) + '/val_penalty(' + str(testi) + ').txt')


        instance_size = rowSizeG * colSizeG
        dim_feat = x.shape[1]
        self.X = x.reshape(-1, rowSizeG, colSizeG, dim_feat)
        self.y = y.reshape(-1, rowSizeG, colSizeG)
        
        num_instances = self.X.shape[0]
        print (num_instances)

        price = np.zeros((num_instances, colSizeG))
        penaltyVector = np.zeros((num_instances, colSizeG)) 
        for num in range(num_instances):
            for i in range(colSizeG):
                price[num, i] = c[i+num*colSizeG]
                penaltyVector [num, i] = penalty[i+num*colSizeG]
        
        self.price = price
        self.penaltyVector = penaltyVector
        self.req = np.tile(req_values, (num_instances, 1))
        print (self.req.shape)
        self.solver = alloy_solver(colSizeG)
        self.sols, self.objs = self._getSols()

    def _getSols(self):
        sols = []
        objs = []
        for i in tqdm(range(len(self.X))):
            sol = self.solver.solve((self.y[i], self.req[i]), self.price[i])
            sols.append(sol)
            objs.append([np.dot(self.price[i], sol)])
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
            tuple: data features (torch.tensor), (weights, req) as tuple of torch.tensor, 
            cost (torch.tensor), optimal solutions (torch.tensor) and objective values (torch.tensor)
        """
        return (
            torch.FloatTensor(self.X[index]),
            (torch.FloatTensor(self.y[index]), torch.FloatTensor(self.req[ index])),
            torch.FloatTensor(self.price[ index]),
            torch.FloatTensor(self.sols[index]),
            torch.FloatTensor(self.objs[index]),
            torch.FloatTensor(self.penaltyVector[ index]),
        )