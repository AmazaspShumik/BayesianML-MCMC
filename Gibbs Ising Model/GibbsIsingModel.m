function [ samples ] = gibbsIsingModel(X, coupleStrength,externalStrength,nSamples, ...
                                       nBurnin, nThin)
% Implements Gibbs on Ising Model (Ising Model is simple example of Markov 
% Random Field). Here we assume that edge potentials are simple 
% 
% Parameters
% ----------
% X: matrix of size (nRows,nColumns)
%    Bianry matrix that consists of {+1,-1} (usually represents
%    preprocessed image)
% 
% externalStrength: float, optional (DEFAULT = 2)
%    Form of clique psi(x_i,y_i). Possible values ['normal','xy']
%  
% coupleStrength: float, optional (DEFAULT = 2)
%    Coupling Strength of Ising Model
% 
% nSamples: integer, optional (DEFAULT = 10)
%    Number of samples
%
% nBurnin: float, optional (DEFAULT = 250)
%    Proportion of samples that are discarded
% 
% nThin: int, optional (DEFAULT = 5)
%    Lag between samples (for thinnig)
%
% Returns
% -------
% samples: Tensor of size (nRows,nCols,nSamples)
%    Samples from posterior distribution p()
%
% Reference
% ---------
% Machine Learning A Probabilistic Perspective (K. Murphy, 2012)

% If optional params are not given initialise to default
if ~exist('externalStrength','var')
    externalStrength = 2;
end
if ~exist('coupleStrength','var')
    coupleStrength = 2;
end
if ~exist('nSamples','var')
    nSamples = 1000;
end
if ~exist('nBurnin','var')
    nBurnin = 250;
end
if ~exist('nThin','var')
    nThin = 10;
end

[nRows,nCols] = size(X);

% inital sample
Xs = X;
samples = zeros(nRows,nCols,nSamples);

for i = 1:(nThin*nSamples + nBurnin)
    
    for r = 1:nRows
        for c = 1:nCols
            sumNeighb = 0;
            if c > 1
                sumNeighb = sumNeighb + Xs(r,c-1);
            end
            if c < nCols
                sumNeighb = sumNeighb + Xs(r,c+1);
            end
            if r > 1
                sumNeighb = sumNeighb + Xs(r-1,c);
            end
            if r < nRows
                sumNeighb = sumNeighb + Xs(r+1,c);
            end          
            E = 2*coupleStrength*sumNeighb;
            
            % joint pdf of observed and unobserved can be changed
            E = E + 2*externalStrength*X(r,c);
            probs = 1./( 1 + exp(-E));
            Xs(r,c) = sign(rand() - probs);
        end
    end
    
    % save samples after burnin and thinning
    if i > nBurnin && mod(i-nBurnin,nThin)==0
       idx = floor((i-nBurnin)/nThin);
       samples(:,:,idx) = Xs;
    end

end

end

