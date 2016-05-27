function [ samples ] = collapsedGibbsLDA(X,nTopics)
% Performs collapsed Gibbs Sample for Latent Dirichlet Allocation
% 
% Parameters
% ----------
% X: Data 
%
% nTopics: int
%    Number of topics
% 
% nSamples: int, optional (DEFAULT = 1000)
%    Number of samples
% 
% nBurnin: int, optional (DEFAULT = 250)
%    Number of samples in the beginning of chain that are discarded
% 
% nThin:
%    Lag between consecutive samples (to avoid autocorrelation)
%
% Returns
% -------
% samples
%
% References
% ----------
% Machine Learning A Probabilistic Perspective (K. Murphy 2012)

if ~exist('nSamples','var')
    nSamples = 1000;
end
if ~exist('nBurnin','var')
    nBurnin = 250;
end
if ~exist('nThin','var')
    nThin = 2;
end


end

