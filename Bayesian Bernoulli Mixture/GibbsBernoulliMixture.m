classdef GibbsBernoulliMixture < handle
    % Superclass for Gibbs Sampler of Bernoulli Mixture
    % Model
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
    % nLag: int, optional (DEFAULT = 10)
    %    Lag between samples ()
    % 
    % priorParams: struct, optional
    %    Parameters of prior distribution
    
    properties(GetAccess='public',SetAccess='private')
        % number of clusters
        nComponents;
        
        % parameters for Gibbs Sampler
        nSamples, nBurnin, nThin;
        saveSamples
    end
    
    methods(Access = 'public')
        
        function obj = GibbsBernoulliMixture(nComponents,nSamples,nBurnin,...
                                             nThin,saveSamples)
           obj.nComponents = nComponents;
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
           if ~exist('saveSamples','var')
               saveSamples = true;
           end
           obj.nSamples = nSamples; obj.nBurnin = nBurnin;
           obj.nThin = nThin; obj.saveSamples = saveSamples;
           
        end
               




    

    
    
end
        
        
    
    
    
    


end

