# -----------------------------
# JIGSAW LINEAR REGRESSION LAB
# -----------------------------            

# ---------------------            
# DIRECTIONS (Phase 1)
# ---------------------

# Working in pairs and using the Driver/Navigator workflow, implement the methods below after carefully reading their docstrings. 
# You should only implement the methods under your GROUP label; for example, if you are a "B" group, you should only implement `LinearRegression1D.gradient`. 
# Switch Driver and Navigator every 5 minutes or every 3 completed lines.  
# You may find it difficult to test your solutions during this phase. 
# That's ok! 

# ---------------------            
# DIRECTIONS (Phase 2)
# ---------------------

# Form groups of 3 so that each group has an "A" member, a "B" member, and a "C" member. 
# Combine your implementations from Phase 1 to form a complete LinearRegression1D class with 5 implemented methods. 
# This is the stage in which you'll be able to informally test your implementations. 
# Do your best to get each function as correct as possible, 


import numpy as np

class LinearRegression1D:
    
    # -----------------------------------
    # GROUP A
    # -----------------------------------
    
    def __init__(self, learning_rate, max_iter, tol):
        pass 
    
    def predict(self, x):
        pass
    
    def score(self, x, y):
        pass
    
    # -----------------------------------
    # GROUP B
    # -----------------------------------
        
    def gradient(self, x_train, y_train):
        pass 

    # -----------------------------------
    # GROUP C
    # -----------------------------------
    
    def gradient_descent(self, x_train, y_train):
        pass 
    
                
                
# ------------------------
# SAMPLE SOLUTION
# ------------------------            

import numpy as np

class LinearRegression1D:
    
    # -----------------------------------
    # GROUP A
    # -----------------------------------
    
    def __init__(self, learning_rate, max_iter,  tol):
        self.beta          = np.random.rand(2)
        self.max_iter      = max_iter
        self.learning_rate = learning_rate
        self.tol           = tol
        
    def predict(self, x):
        return self.beta[1]*x + self.beta[0]
    
    def score(self, x, y):
        y_pred = self.predict(x)
        return ((y - y_pred)**2).mean()
    
    # -----------------------------------
    # GROUP B
    # -----------------------------------
    
    def gradient(self, x_train, y_train):
        y_pred = self.predict(x_train)
        grad = 2*np.array([
             (y_pred - y_train).mean(),
            ((y_pred - y_train)*x_train).mean()
        ])
        return grad
    
    # -----------------------------------
    # GROUP C
    # -----------------------------------
    
    def gradient_descent(self, x_train, y_train):
        i = 0
        while i <= self.max_iter:
            grad = self.gradient(x_train, y_train)
            if np.abs(grad).sum() < self.tol: 
                break
 
            self.beta -= self.learning_rate*grad
            i += 1
           