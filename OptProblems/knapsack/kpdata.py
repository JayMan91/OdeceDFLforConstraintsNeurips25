"""
Synthetic data for knapsack problem
"""
import numpy as np

def genWeights (num_data, num_features, num_items, capacity_ratio = np.ones(1), dim=1, 
                deg=1, noise_width=0, fixed_cost = False, seed=135):
    """
    Generate synthetic data for knapsack problem
    Args:
        num_data (int): number of data points
        num_features (int): number of features
        num_items (int): number of items
        capacity_ratio (numpy ndarray): shape same as dim, ratio of capacity
        dim (int): dimension of multi-dimensional knapsack
        deg (int): data polynomial degree
        noise_width (float): half width of data random noise
        fixed_cost (bool): whether to same cost vector for all data points
        seed (int): random seed
    Returns:
        X (np.ndarray) shape (num_data, num_features): feature matrix
        w (np.ndarray) shape (num_data, dim, num_items): weights of knapsack items
        costs (np.ndarray) shape (num_data, num_items): value of knapsack problem
        capacity (np.ndarray) shape (num_data, dim): capacity of knapsack
    """
    # set seed
    rnd = np.random.RandomState(seed)
    # number of data points
    n = num_data
    # dimension of features
    p = num_features
    # dimension of multi-dimensional knapsack problem
    d = dim
    # number of items
    m = num_items
    # generate features
    X = rnd.uniform(0, 1, (n, p))
    B = rnd.binomial(1, 0.5, (d, m, p))
    # full_capacity = np.sum(  1 + ((3 + B/np.sqrt(p))**deg)/(3.5 ** deg), axis=(1, 2))
    
    full_capacity = np.sum (B, axis=(-1, -2))
    # print (B)
    # print (full_capacity)
    capacity = capacity_ratio * full_capacity
    # noise
    epislon = rnd.uniform(1 - noise_width, 1 + noise_width, (n, d))
    capacity =  np.ceil (10*capacity * epislon)

    # generate weights
    w = np.zeros((n, d, m))
    for i in range(n):
        weights = np.sum(B * X[i], axis=2)
        weights /= np.sqrt(p) 
        weights = 1 + (3 + weights)**deg
        # rescale
        weights /= 3.5 ** deg
        # noise
        epislon = rnd.uniform(1 - noise_width, 1 + noise_width, m)
        weights *= epislon
        # print (weights)
        # convert into int
        weights = np.ceil(100*weights)
        w[i] = weights/10
        # print ("weights: ", weights)   
    # print ("capacity: ", capacity) 
    # generate costs
    X = X.astype(np.float64)
    w = w.astype(np.float64)
    if fixed_cost:
        costs = rnd.gumbel(100, 20, size=(m))
        costs = np.clip(costs, 5, None)
        costs = np.tile(costs, (n, 1))
    else:
        costs = rnd.gumbel(100, 20, size=(n, m))
        costs = np.clip(costs, 5, None)
    return X, w, costs, capacity

def genCapacity (num_data, num_features, num_items,  dim=1, 
                deg=1, noise_width=0, ratio = 0.2, fixed_cost = False, seed=135):
    """
    Generate synthetic data for knapsack problem
    Args:
        num_data (int): number of data points
        num_features (int): number of features
        num_items (int): number of items
        dim (int): dimension of multi-dimensional knapsack
        deg (int): data polynomial degree
        noise_width (float): half width of data random noise
        fixed_cost (bool): whether to same cost vector for all data points
        seed (int): random seed
    Returns:
        X (np.ndarray) shape (num_data, num_features): feature matrix
        w (np.ndarray) shape (num_data, dim, num_items): weights of knapsack items
        costs (np.ndarray) shape (num_data, num_items): value of knapsack problem
        capacity (np.ndarray) shape (num_data, dim): capacity of knapsack
    """
    # set seed
    rnd = np.random.RandomState(seed)
    # number of data points
    n = num_data
    # dimension of features
    p = num_features
    # dimension of multi-dimensional knapsack problem
    d = dim
    # number of items
    m = num_items
    weights = rnd.choice(range(1000, 2000), size=(n , d, m)) / 1000
    weights = np.round(weights, 2).astype(np.float64)
    # generate features
    X = rnd.uniform(0, 1, (n, p))
    B = rnd.binomial(1, 0.5, (d, p))

    capacity = np.zeros((n, d), dtype=int)
    # generate capacity
    ### Weights are between 10 and 20
    ### Clip the weights between 50 and 5*num_items
    ## This to ensure atleast 3 items can be picked and not all items can be picked
    for i in range(n):
        values = (np.dot(B, X[i].reshape(p, 1)).T / np.sqrt(p) + 3) ** deg + 1
        # rescale
        # values *= 5
        values /= 3.5 ** deg
        # noise
        epislon = rnd.uniform(1 - noise_width, 1 + noise_width,  d)
        values *= epislon
        X[i] *= 0.5*m 
        values *=  5*m
        values = np.clip(values, None, 5*m)
        values = np.ceil(values)
        capacity[i,:] = values/10
        # print ("New: ", values)
    capacity = capacity.astype(np.float64)
    X = np.around(X)
    X = X.astype(np.float64)

    if fixed_cost:
        costs = rnd.gumbel(100, 20, size=(m))
        costs = np.clip(costs, 5, None)
        costs = np.tile(costs, (n, 1))
    else:
        costs = rnd.gumbel(100, 20, size=(n, m))
        costs = np.clip(costs, 5, None)

    return X, weights , costs, capacity

if __name__ == "__main__":
    X, w, costs, capacity = genCapacity(num_data= 8, num_features=4, num_items=12, dim= 5,
     deg=6, noise_width=0.1, seed=135)
    print ("Dimension of X: ", X.shape)
    print ("Dimension of w: ", w.shape)
    print ("Dimension of costs: ", costs.shape)
    print ("Dimension of capacity: ", capacity.shape)

 