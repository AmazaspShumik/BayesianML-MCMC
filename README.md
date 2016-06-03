# Bayesian Machine Learning using Monte Carlo Markov Chain
This repository contains implementation of several Machine Learning Models using MCMC. This repository does not conatin production quality code, however I hope that somebody will find it useful for learning & development of their own models. 

## Bernoulli Mixture Model: Clustering Digits with Gibbs Sample
In this example we comapre collapsed and vanilla Gibbs samplers applied to image clustering. In vanilla Gibbs we sample by consecutively computing full conditionals for each random variable, while in collapsed Gibbs we integrate out some random variables analytically and perform standard Gibbs sample on the ones that were not integrated out. So collapsed Gibbs has advantage of sampling from lower dimensional space, it also should have smaller variance (see Rao-Blackwell theorem). However sometimes sampling from full conditionals after integrating out can be costly, also some conditional independence properties that exist in full model are lost after analytical integration and therefore collapsed Gibbs is more difficult to parallelize.
In this example we used subsample from digit recognition competition in kaggle that contains ony three digits 1,3,6. In figure below you can see 5 samples of posterior means from collapsed and vanilla Gibbs samplers.

![alt tag](https://github.com/AmazaspShumik/BayesianML-MCMC/blob/master/Gibbs%20Bernoulli%20Mixture/meanSamples.jpg)

As we can see from plot of log-likelihoods collapsed sampler converges faster (measured in number of samples) than vanilla one, however it should be noted that each iteration of collapsed sampler requires more time. Both samplers converge to almost the same value of log-likelihood.
![alt tag](https://github.com/AmazaspShumik/BayesianML-MCMC/blob/master/Gibbs%20Bernoulli%20Mixture/logLikePlot.jpg)

## Ising Model : Image Denoising Example
In the example below we show how Ising Model can be used for Image Denoisning Tasks (image in this example is taken from Frank Wood's course in Columbia University). We use Gibbs Sampling to obtain samples from posterior of hidden variables.

![alt tag](https://github.com/AmazaspShumik/BayesianML-MCMC/blob/master/Gibbs%20Ising%20Model/imageDenoisingDemo.jpg)

In original image 9.73% of pixels were flipped to produce noisy image, after denoising only 1% of pixels remained flipped.
![alt tag](https://github.com/AmazaspShumik/BayesianML-MCMC/blob/master/Gibbs%20Ising%20Model/proportionWrongPixels.jpg)

## Latent Dirichlet Distribution: Topic Modelling




