# -----------------------------
# JIGSAW LINEAR REGRESSION LAB
# -----------------------------            

# ---------------------            
# DIRECTIONS (Phase 1)
# ---------------------

# Working in pairs and using the Driver/Navigator workflow, implement the
# methods below after carefully reading their docstrings. 
# You should only implement the methods under your GROUP label; for example, if 
# you are a "B" group, you should only implement `LinearRegression1D.gradient`. 
# Switch Driver and Navigator every 5 minutes or every 3 completed lines.  
# You may find it difficult to test your solutions during this phase. 
# That's ok! 

# ---------------------            
# DIRECTIONS (Phase 2)
# ---------------------

# Form groups of 3 so that each group has an "A" member, a "B" member, and a
# "C" member. 
# Combine your implementations from Phase 1 to form a complete 
# LinearRegression1D class with 5 implemented methods. 
# This is the stage in which you'll be able to informally test your 
# implementations. 
# Do your best to write each function as correctly as possible. 
# Once you think you've got it, begin to test your implementation by running 
# the cells in `handout.ipynb`. 

# ------------------------
# SAMPLE SOLUTION
# ------------------------            

import numpy as np
import warnings


class LinearRegression1D:
    
    '''
    A simple model of a linear relationship between a vector x of predictor data and a vector y of target data. 
    
    INSTANCE VARIABLES: 
    
        learning_rate float64: the positive coefficient of the gradient in the gradient descent step. Small learning rate can lead to more accurate results but can also lead to longer runtimes. Specified by user in LinearRegression1D.__init__(). 

        max_iter int: the maximum number of iterations of the gradient descent update to perform before stopping. Termination due to reaching max_iter can result in poor model fit. Specified by user in LinearRegression1D.__init__().
        
        tol float64: the numerical threshold at which the gradient is considered to be "close enough to 0" to terminate gradient descent. Specified by user in LinearRegression1D.__init__(). 
        
        beta np.ndarray of shape (2,): the coefficients of the linear relationship. beta[0] represents the intercept and beta[1] represents the slope. Initialized randomly during LinearRegression1D.__init__(). 
    
    '''    
    
    # -----------------------------------
    # GROUP A
    # -----------------------------------
    
    def __init__(self, learning_rate, max_iter,  tol):
        self.beta          = np.random.rand(2)
        self.max_iter      = max_iter
        self.learning_rate = learning_rate
        self.tol           = tol
        
    def predict(self, x):
        """
        Form predictions from predictor data using a 1D linear model. 
        
        ARGS: 
            x np.ndarray of shape (m,), where m is an integer: an array of predictor data. 
            
        RETURN: 
            ŷ np.ndarray of shape (m,): an array of model predictions. In the 1D linear model, prediction corresponding to the ith entry of x is 
            
            ŷᵢ = β₀ + β₁xᵢ
         
        """
        return self.beta[1]*x + self.beta[0]
    
    def mse(self, x, y):
        """
        Evaluate the mean-square error of a 1D linear model on a supplied set of predictor and target data. 
        
        ARGS: 
            x np.ndarray of shape (m,), where m is an integer: an array of predictor data. 
            
            y np.ndarray of shape (m,): an array of target data. 
            
        RETURN: 
            MSE, the mean-square error associated with using the model to predict y from x. The MSE is 
            
            1/n*Σᵢ(ŷᵢ - yᵢ)²
            
            where ŷ is calculated from LinearRegression1D.predict(x)
        
        """
        y_pred = self.predict(x)
        return ((y - y_pred)**2).mean()
    
    # -----------------------------------
    # GROUP B
    # -----------------------------------
    
    def gradient(self, x_train, y_train):
        """
        Calculate the gradient of the mean-square error (MSE) when using x_train to predict y_train with respect to the model parameters beta. 
        
        ARGS: 
            x_train np.ndarray of shape (m,), where m is an integer: an array of predictor data. 
            
            y_train np.ndarray of shape (m,): an array of target data.   

        RETURN: 
            grad np.ndarray of shape (2,), the gradient of the MSE with respect to the parameters beta[0] and beta[1]. grad[0] is the derivative of the MSE with respect to beta[0], and grad[1] is the derivative of the MSE with respect to beta[1]. NOTE: these can be calculated by hand. 
        
        """
        y_pred = self.predict(x_train)
        grad = 2*np.array([
             (y_pred - y_train).mean(),
            ((y_pred - y_train)*x_train).mean()
        ])
        return grad
    
    # -----------------------------------
    # GROUP C
    # -----------------------------------
    
    def fit(self, x_train, y_train):
        """
        Fit a LinearRegression1D model to data using a simple gradient descent algorithm with constant learning rate. 
        
        ARGS: 
            x_train np.ndarray of shape (m,), where m is an integer: an array of predictor data. 
            
            y_train np.ndarray of shape (m,): an array of target data.  
            
        RETURN: 
            None   
            
        NOTES: 
        
            In each step of gradient descent, update beta according to the formula 
            
            beta <- beta - learning_rate*grad
            
            where grad is calculated by LinearRegression1D.gradient() and beta and learning_rate are instance variables. 
            
            Implement two stopping criteria: 
            
                1. If the maximum number of iterations is reached (according to max_iter), then stop and raise a warning to the user that the maximum number of iterations was reached and that the model fit may not be optimal. 
                2. If the sum of the absolute values of the entries of the gradient is below self.tol, then stop without a warning. 
        """
        i = 0
        while i <= self.max_iter:
            grad = self.gradient(x_train, y_train)
            if np.abs(grad).sum() < self.tol: 
                break
 
            self.beta -= self.learning_rate*grad
            i += 1
        
        if i > self.max_iter: 
            warnings.warn("Maximum number of iterations reached, model fit may be poor.")
           