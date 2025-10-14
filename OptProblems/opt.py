MINIMIZE = 1
MAXIMIZE = -1

from abc import abstractmethod
class optSolver():
    @abstractmethod   
    def __init__(self, modelSense = MINIMIZE):

        self.modelSense = modelSense

    @abstractmethod
    def solve(self,  param_constraints, param_objective):
        raise NotImplementedError

    @abstractmethod
    def check_feasibility(self, param_constraints, sol):
        raise NotImplementedError

    @abstractmethod
    def evaluate_solution(self, param_objective, sol):
        raise NotImplementedError