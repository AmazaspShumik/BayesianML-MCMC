function [ samples ] = GibbsIsingModel(X, coupleStrength,xyCliqueForm,nSamples, ...
                                       nBurnin, nThin)
% Implements Gibbs on Ising Model (Ising Model is simple example of Markov 
% Random Field). Here we assume that edge potentials are simple 
% 
% Parameters
% ----------
% X: matrix of size (nRows,nColumns)
%    Bianry matrix (usually represents image)
% 
% xyCliqueForm: string, optional (DEFAULT = 'normal')
%    Form of clique psi(x_i,y_i). Possible values ['normal','xy']
%  
% coupleStrength: float, optional (DEFAULT = 2)
%    Coupling Strength of Ising Model
% 
% nSamples: integer, optional (DEFAULT = 1000)
%    Number of samples
%
% nBurnin: float, optional (DEFAULT = 0.25)
%    Proportion of samples that are discarded
% 
% nThin: int, optional (DEFAULT = 10)
%    Lag between samples (for thinnig)
%
% Returns
% -------
% samples

% If optional params are not given initialise to default
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
Xs = ones(nRows,nCols);
yMean = sum(sum(Y)) / (nRows*nCols);
Xs(X < yMean) = -1;
samples = zeros(nRows,nCols,nSamples);

for i = 1:(nThin*nSamples + nBurnin)
    
    sumNeighb = 0;
    for r = 1:nRows
        for j = 1:nCols
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
            if xyCliqueForm == 'normal'
                E = E - lognpdf(X(r,s),Xs(r,c),sigma);
                E = E + lognpdf(X(r,s),-Xs(r,c),sigma);
            elseif xyCliqueForm == 'xy'
                E = E - log()
            end
                
            probs = 1./( 1 + exp(-E));
            Xs(r,c) = sign(mnrnd(1,[probs,1-probs]) - 0.5);
        end
    end
    
    % save sample 
    if i > nBurnin && mod(i-nBurnin,nThin)==0
       idx = floor((i-nBurnin)/nThin);
       samples(r,c,idx) = Xs;
    end

end

end

