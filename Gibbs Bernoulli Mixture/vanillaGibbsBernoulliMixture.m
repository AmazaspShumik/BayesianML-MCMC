function [ muSamples, clusters, logLike ] = vanillaGibbsBernoulliMixture(X,nComponents,nSamples,...
                                                    nBurnin,nThin,priorParams,logLikeCompute)
% Bernoulli Mixture Model implemented with vanilla Gibbs Sample.
% 
% Parameters
% ----------
% nComponents: integer
%    Number of components in mixture model
%
% nSamples: integer, optional (DEFAULT = 10000)
%    Number of samples
%
% nBurnin: float, optional (DEFAULT = 0.25)
%    Proportion of samples that are discarded
% 
% nThin: int, optional (DEFAULT = 10)
%    Lag between samples (for thinnig)
% 
% priorParams: struct, optional
%    Parameters of prior distribution
%    .latentDist : prior for latent distribution [1,nComponents]
%    .muBeta : shape parameter for mean prior [nComponents,nFeatures]
%    .muGamma : shape parameter for mean prior [nComponents,nFeatures]
%
% logLikeCompute: bool, optional (DEFAULT = true)
%    If true computes log-likelihood of data at each iteration
%
% Returns
% -------
% muSamples: Tensor of size (nComponents,nFeatures,nSamples)
%      Samples from means of clusters
%
% clusters: matrix of size (nDataSamples,nSamples)
%      Cluster ids for each samples (Be aware of possible label switch)
%
% logLike : Vector of size (1,nSamples*nThin + nBurnin)
%      Vector of loglikelihoods

% handle optional parameters
if ~exist('nSamples','var')
    nSamples = 10000;
end          
if ~exist('nBurnin','var')
    nBurnin  = 2500;
end
if ~exist('nThin','var')
    nThin    = 10;
end
if ~exist('logLikeCompute','var')
    logLikeCompute = true;
end
    

% number of datapoints & dimensionality
[nDataSamples,nFeatures] = size(X);
if ~exist('priorParams','var')
    latentDist = 1 + rand(1,nComponents);
    muBeta    = 1 + rand(nComponents,nFeatures);
    muGamma   = 1 + rand(nComponents,nFeatures);
else
    latentDist = priorParams.latentDist;
    muBeta    = priorParams.muBeta;
    muGamma   = priorParams.muGamma;
end

% start Gibbs Sampler & allocate memory
muSample   = betarnd(muBeta,muGamma);
prSample   = dirchrnd(latentDist);
logResps   = zeros(nDataSamples,nComponents);
muSamples  = zeros(nComponents,nFeatures,nSamples);
logLike    = zeros(1,nSamples*nThin + nBurnin);
clusters   = zeros(nDataSamples,nSamples);
            
for i = 1:(nSamples*nThin+nBurnin)
                
    % compute responsibilities for sampling from latent
    for k = 1:nComponents
        logResps(:,k) = binologpdf(X,muSample(k,:));
        logResps(:,k) = logResps(:,k) + log(prSample(k));
    end
    resps = exp(bsxfun(@minus,logResps,logsumexp(logResps,2)));
    resps = bsxfun(@rdivide,resps,sum(resps,2));
                
    % sample p( z_i | X, Z_{-i}, mu, pr )
    latentSample = sparse(mnrnd(1,resps,nDataSamples));
    
    % reuse log responsibilities for calculation of log probs
    if logLikeCompute
        logLike(i) = sum(sum(latentSample.*logResps));
    end
    Nk = sum(latentSample,1);
    Xweighted = latentSample'*X;
    IXweighted = -bsxfun(@minus,Xweighted,Nk');

    % sample p( pr | X, Z, mu_{1:k} )
    prSample = dirchrnd( latentDist + Nk);
    
    % sample p( mu_k | X, Z, mu_{-k}, pr )
    muSample = betarnd(muBeta + Xweighted,muGamma + IXweighted);
                
    if i > nBurnin && mod(i-nBurnin,nThin)==0
       % accept sample after burnin & thinning
       idx = floor((i-nBurnin)/nThin);
       muSamples(:,:,idx) = muSample;
       [Max,clusterIndex] = max(latentSample,[],2); 
       clusters(:,idx)    = clusterIndex;
    end
end

end

