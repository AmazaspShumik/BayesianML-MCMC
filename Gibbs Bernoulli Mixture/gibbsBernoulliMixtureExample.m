% This script illustrates use of Bernoulli Mixture 
%
% Data used in example is random sample from subset of data provided
% in digit recognition competition in kaggle
% We use only three digits : 1,6,3 (100 data points for each of them)

% load data 
X = csvread('digits_examples.csv');

% parameters of model
nComponents = 3;
nSamples    = 10;
nBurnin     = 20;
nThin       = 2;
logLikeCompute = true; % collapsedGibbs should be faster without this

% vanilla Gibbs Sample
[vMuSamples,vClusters,vLogLike] = vanillaGibbsBernoulliMixture(X,nComponents,...
                                  nSamples,nBurnin,nThin,logLikeCompute);
% collapsed Gibbs Sample
[cMuSamples,cClusters,cLogLike] = collapsedGibbsBernoulliMixture(X,nComponents,...
                                  nSamples,nBurnin,nThin,logLikeCompute);

% show all three components from last sample 
for j = 1:nComponents
    figure(j)
    imshow(reshape(sign(vMuSamples(j,:,nSamples)-0.5),28,28)')
    title('Sample component mean, vanilla Gibbs')
    figure(j+nComponents)
    imshow(reshape(sign(cMuSamples(j,:,nSamples)-0.5),28,28)')
    title('Sample component mean, collapsed Gibbs')
end

% plot log-likelihoods, show that collapsedGibbs converges faster
figure(2*nComponents+1)
plot(vLogLike,'b-')
hold on
plot(cLogLike,'r-')
xlabel('Iterations')
ylabel('log-likelihood')
legend('vanillaGibbs','collapsedGibbs','Location','southeast')
title('Convergence coparison for Gibbs Sampler: collapsed vs vanilla')

    