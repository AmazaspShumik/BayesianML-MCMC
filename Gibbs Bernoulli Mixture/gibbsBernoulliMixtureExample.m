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
nBurnin     = 100;
nThin       = 2;
logLikeCompute = true; % collapsedGibbs should be faster without this

% vanilla Gibbs Sample
[vMuSamples,vClusters,vLogLike] = vanillaGibbsBernoulliMixture(X,nComponents,...
                                  nSamples,nBurnin,nThin,logLikeCompute);
% collapsed Gibbs Sample
[cMuSamples,cClusters,cLogLike] = collapsedGibbsBernoulliMixture(X,nComponents,...
                                  nSamples,nBurnin,nThin,logLikeCompute);

% show all three components from last sample
im = ones(178,200);
nImageSamples = nSamples/2;
rowMin = 1; rowMax = 28;
for n = 1:nImageSamples
    colMin = 1; colMax = 28;
    for j = 1:nComponents
        % image from vanilla Gibbs sampler
        vim = reshape(sign(vMuSamples(j,:,n)-0.5),28,28)';
        im(rowMin:rowMax,colMin:colMax) = vim;
        % image from collapsed Gibbs sampler
        cim = reshape(sign(cMuSamples(j,:,n)-0.5),28,28)';
        shift = nComponents*28+20;
        im(rowMin:rowMax,colMin+shift:colMax+shift) = cim;
        colMin = colMin + 30;
        colMax = colMax + 30;
    end
    rowMin = rowMin + 30;
    rowMax = rowMax + 30;
end
imshow(im);
title('Posterior Mean Samples: vanilla vs collapsed Gibbs')

% plot log-likelihoods, show that collapsedGibbs converges faster
figure(2*nComponents+1)
plot(vLogLike(1:20),'b-','linewidth',3)
hold on
plot(cLogLike(1:20),'r-','linewidth',3)
xlabel('Iterations')
ylabel('log-likelihood')
legend('vanillaGibbs','collapsedGibbs','Location','southeast')
title('Convergence coparison for Gibbs Sampler: collapsed vs vanilla')

    