import numpy as np
import pandas as pd

class GLM(object):
    
    def __init__(self, X = np.array([]), y = np.array([]), reg = 0, name = 'GLM'):
        
        self.name = name
        self.X = np.vstack([np.ones((X.shape[0], )), np.array(X).T]).T
        self.y = np.array(y)
        self.beta = np.zeros(self.X.shape[1])
        self.reg = reg
    
    def train(self, output = False):
        
        
        beta_ = np.inf
        count = 0
        
        while max(abs(self.beta - beta_)) > float('1e-6'):
            pi = self._probabilities() # ndarray
            var = self._var() # ndarray of variances
            VX = self.X*(var.reshape(-1, 1))
            XVX_lambda = np.dot(self.X.T, VX) + self.reg * np.identity(self.X.shape[1])
            
#             VXbeta = np.dot(VX, self.beta)
            H_plus_g_beta = np.matmul(XVX_lambda, self.beta) + np.matmul(self.X.T, self.y - pi) - self.reg * self.beta
            beta_ = self.beta
            self.beta = np.linalg.solve(XVX_lambda, H_plus_g_beta)
            
            
        return self.beta
     
    def predict(self, X_test):
        
        X_test = np.vstack([np.ones((X_test.shape[0], )), np.array(X_test).T]).T

        return self._invLink(np.matmul(X_test, self.beta))
    
    def _invLink(self):
        pass
    
    def _var(self):
        pass
    
    def _probabilities(self):
        Eta = np.matmul(self.X, self.beta)
        
        return self._invLink(Eta)
    
class LogisticRegression(GLM):
    
    
    def _invLink(self, eta):
        
        return 1 / (1 + np.exp(-eta))
    
    def _var(self):
        prob = self._probabilities()
        return prob * (1 - prob)
    

class PoissonRegression(GLM):
    
    
    def _invLink(self, eta):
        
        return np.exp(eta)
    
    def _var(self):
        
        return  self._probabilities()