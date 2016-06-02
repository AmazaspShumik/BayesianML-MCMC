# Bayesian Machine Learning using Monte Carlo Markov Chain
This repository contains implementation of several Machine Learning Models using MCMC. This repository does not conatin production quality code, however I hope that somebody will find it useful for learning & development of their own models. 


## Ising Model : Image Denoising Example
In the example below we show how Ising Model can be used for Image Denoisning Tasks (image in this example is taken from Frank Wood's course in Columbia University). We use Gibbs Sampling to obtain samples from posterior of hidden variables.

 ![alt tag](https://github.com/AmazaspShumik/BayesianML-MCMC/blob/master/Gibbs%20Ising%20Model/imageBeforeDenoising.jpg)
 ![alt tag](https://github.com/AmazaspShumik/BayesianML-MCMC/blob/master/Gibbs%20Ising%20Model/imageDenoisedSample.jpg)

As you can see from posterior sample above most of the noise is removed. In original image 9.73% of pixels were flipped to produce noisy image, after denoising only 1% of pixels reamined flipped.
![alt tag](https://github.com/AmazaspShumik/BayesianML-MCMC/blob/master/Gibbs%20Ising%20Model/proportionWrongPixels.jpg)






