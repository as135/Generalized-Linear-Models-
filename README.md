# Generalized Linear Models
I've coded up a generalized linear model class for solving problems with responses whose means are distributed via the exponential family of distributions. So far I've included Logistic Regression, Poisson Regression, and Linear Regression, with more to come. The solvers used initially were Newtonian (Iteratively Reweighted Least Squares), as of 5/18 added support for various gradient descents in the jupyter notebook. 

Generalized Linear Models are fascinating with their combination of predictive power and mathematical elegance. The fact that they can be solved efficiently using second order methods with guaranteed convergence is especially attractive in the age of neural networks. On that note, it is certainly worthwhile to consider NNs to be special cases of GLMs in which the hidden layers are attempting to learn the best representation for the inputs of the final GLM layer. 

I am also interested in coding up efficient Bayesian posterior estimators for each of the models to shore up their capability. 

# Update 5/8

You can now assign L2 regularization parameter.

# Update 5/10 

You can now use Linear Regression

# Update 5/18

3 Variants of Gradient Descent (batch, stochastic, and mini-batch) have been implemented for GLMs in the testing notebook


#   TODO

1. Add support for Bayesian methods  (MCMC estimation of posterior by Metropolis Hastings) and Gaussian Process Regression
2. Add Support for Softmax regression in Logistic Case
3. Add support for further distributions (Gamma, Beta, Binomial, Quasi-Poisson)
4. Add support for Kernel Methods
5. Implement Lp norm and Elastic Net  
6. Add support for GLMMs (fixed and random effects)  
7. Variational Inference for Graphical Models (Variational Autoencoder to find latent space)

