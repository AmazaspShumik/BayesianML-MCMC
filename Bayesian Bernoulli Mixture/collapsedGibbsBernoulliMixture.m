function [muSamples, clusters,logLike] = collapsedGibbsBernoulliMixture(X,nComponents,nSamples,...
                                                    nBurnin, nThin,priorParams,logLikeCompute)
% Bayesian Bernoulli Mixture Model with collapsed Gibbs Sample.
% Collapsed Gibbs Sample converges with smaller number of iterations
% and has smaller variance than standard Gibbs Sample
% (see Rao-Blackwell theorem), however each sample in collapsed Gibbs is 
% more expensive.
% In MATLAB due to the speed of vectorized operations vanilla Gibbs can 
% be sometimes faster than collapsed Gibbs (since collapsed uses for 
% loop to iterate through data samples)
%
% Parameters
% ----------
% nComponents: integer
%     Number of components in mixture model
% 
% nSamples: integer, optional (DEFAULT = 10000)
%     Number of samples
%
% nBurnin: float, optional (DEFAULT = 0.25)
%    Proportion of samples that are discarded
% 
% nThin: int, optional (DEFAULT = 10)
%    Lag between samples (for thinnig)
% 
% priorParams: struct, optional
%    Parameters of latent variable distribution (after integrating out pr)
%    .latentPrior - prior for latent variable
%    .muBeta      - shape parameter for Beta prior of mean
%    .muGamma     - shape parameter for Beta prior of mean
%
% logLikeCompute: bool, optional (DEFAULT = true)
%    If true computes log-likelihood of data at each iteration
%    IMPORTANT NOTE: This makes sampling slower since requires sampling 
%    mixing probabilities and means (which otherwise are not sampled)
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

[nDataSamples,nFeatures] = size(X);
if ~exist('priorParams','var')
    muBeta       = 1+1*rand(1);
    muGamma      = 1+1*rand(1);
    latentPrior  = 1+1*rand(1);    
else
    muBeta  = priorParams.muBeta;
    muGamma = priorParams.muGamma;
    latentPrior = priorParams.latentPrior;
end

% generate initial assignments of latent variables
latentVar = mnrnd(1,ones(1,nComponents)/nComponents,nDataSamples);
clusters  = zeros(nDataSamples,nSamples);
muSamples  = zeros(nComponents,nFeatures,nSamples);
logLike   = zeros(1,nSamples*nThin + nBurnin);

for i = 1:(nSamples*nThin+nBurnin)
    
    Nk = sum(latentVar,1);
    Ck = latentVar'*X;
    % generate random permuatation 
    for j = 1:nDataSamples
        
        % remove sufficient stats for current point
        Nk_j = Nk - latentVar(j,:);
        Ck_j = Ck - kron(X(j,:),latentVar(j,:)');
        
        % compute log p(x | Z_{-i}, z_{i} = k) [not normalised]
        muBetaPost = muBeta + Ck_j;
        muGammaPost = muGamma + bsxfun(@minus,Nk_j',Ck_j);
        logSucces = bsxfun(@times,log(muBetaPost),X(j,:));
        logFail   = bsxfun(@times,log(muGammaPost),1 - X(j,:));
        logJoint  = log( bsxfun(@plus,Nk_j',muGamma + muBeta));
        logPx     = sum(logSucces + logFail,2)- nComponents*logJoint;
        
        % compute log p(z_{i} | Z_{-i}, X)
        logPz     = log(latentPrior + Nk_j) + logPx';
        Pz        = exp(bsxfun(@minus,logPz,logsumexp(logPz,2)));
        Pz        = bsxfun(@rdivide,Pz,sum(Pz,2));
        
        % sample from multinoulli distribution
        latentVar(j,:) = mnrnd(1,Pz,1);
        
        %update sufficient stats
        Nk = Nk_j + latentVar(j,:);
        Ck = Ck_j + kron(X(j,:),latentVar(j,:)');       
    end
    
    % Note that in collapsed Gibbs we need to sample only latent variable,
    % however if we want to compute loglikelihood we will need to sample
    % success probabilities and mixing probabilities
    if logLikeCompute
        % sample mixing probs
        prSample = dirchrnd( latentPrior + Nk);
        % sample means
        muBetaPost  = muBeta + Ck;
        muGammaPost = muGamma + bsxfun(@minus,Nk_j',Ck_j);
        muSample  = betarnd(muBetaPost,muGammaPost);
        % compute log-likelihood
        for k = 1:nComponents
            componentIndex = latentVar(:,k)==1;
            logLikeK = sum( binologpdf(X(componentIndex,:),muSample(k,:)));
            logLike(i) = logLike(i) + logLikeK + log(prSample(k))*sum(componentIndex);
        end
        % if after burnin & thinning applied save mean sample
        if i > nBurnin && mod(i-nBurnin,nThin)==0
            idx = floor((i-nBurnin)/nThin);
            muSamples(:,:,idx) = muSample;
        end
    end
        
    if i > nBurnin && mod(i-nBurnin,nThin)==0
        
       % accept sample after burnin & thinning
       idx = floor((i-nBurnin)/nThin);
       [Max,clusterIndex] = max(latentVar,[],2); 
       clusters(:,idx)    = clusterIndex;
       % sample means from posterior
       if logLikeCompute ~= true
           muBetaPost  = muBeta + Ck;
           muGammaPost = muGamma + bsxfun(@minus,Nk_j',Ck_j);
           muSample  = betarnd(muBetaPost,muGammaPost);
           muSamples(:,:,idx) = muSample;
       end
    end  
end


