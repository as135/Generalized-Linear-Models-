# Generalized Linear Models
I've coded up a generalized linear model class for solving problems with responses whose means are distributed via the exponential family of distributions. So far I've included Logistic Regression and Poisson Regression, with more to come. The solvers used are Newtonian (Iteratively Reweighted Least Squares); hoping to add more in the future.

# Update 5/8

You can now assign L2 regularization parameter.

# Update 5/10 

You can now use Linear Regression





#   TODO

1. Add support for kernel methods
2. Add Support for Softmax regression in Logistic Case
3. Add support for further distributions (Gamma, Beta, Binomial, Quasi-Poisson)
4. Add support for Bayesian methods  (MCMC)
5. Implement gradient descent
     5.a Implement Lp norm and Elastic Net
