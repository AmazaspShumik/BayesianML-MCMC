function [ samples, loglike] = coll_gibbs_lda(Ws,Ds,nTopics, alpha, gamma,...
                                         nSamples,nBurnin,nThin)
% Performs collapsed Gibbs Sample for Latent Dirichlet Allocation
% 
% Parameters
% ----------
% Ws: cell array of size (1,nCorpus)
%             Word encoding of whole corpus
%
% Ds: vector of size (1,nCorpus)
%             Topic encoding of whole corpus
%
% nTopics: int
%             Number of topics
%
% alpha: float, optional (DEFAULT = 2)
%             Concentration parameter for dirichlet prior of topic distribution
%
% gamma: float, optional (DEFAULT = 2)
%             Concentration parameter for dirichlet prior of word distribution
% 
% nSamples: int, optional (DEFAULT = 1000)
%             Number of samples
% 
% nBurnin: int, optional (DEFAULT = 250)
%             Number of samples in the beginning of chain that are discarded
% 
% nThin: int, optional (DEFAULT = 2)
%             Lag between consecutive samples (to avoid autocorrelation)
%
% Returns
% -------
% samples: struct with fields ('topicAssignment','nDocumentTopic','nWordTopic','nTopic')
%            Samples from posterior distribution of latent variable  
%           
%       - topicAssignment(:,sample_i):  vector of size (nCorpus,1)
%               Topic assignment for each word in corpus in i-th sample
%         
%       - nDocumentTopic(:,:,sample_i): matrix of size (nDocs,nTopics)
%               nDocumentTopic(d,k,sample_i) - number of words assigned 
%               to topic k in document d in i-th sample
% 
%       - nWordTopic(:,:,sample_i): matrix of size (nVocab,nTopics)
%              nWordTopic(v,k,sample_i) -  number of times word v was 
%              assigned to topic k in i-th sample
% 
%       - nTopic(:,sample_i): vector of size (nTopics,1)
%              Number of words assigned to each topic in i-th sample
% 
%       - loglike(sample_i): float
%              Joint log-likelihood for i-th sample
%             
% References
% ----------
% Machine Learning A Probabilistic Perspective (K. Murphy 2012)

% handle default values of variables that user did not define
if ~exist('nSamples','var')
    nSamples = 5;
end
if ~exist('nBurnin','var')
    nBurnin = 20;
end
if ~exist('nThin','var')
    nThin = 2;
end
if ~exist('alpha','var')
    alpha = 2;
end
if ~exist('gamma','var')
    gamma = 2;
end

%---------- Precompute Required Statistics & Initialise Chain -----------

% Corpus size, number of docs, vocabulary size
nCorpus = length(Ws);
nDocs   = max(Ds);
nVocab  = max(Ws);

% initial topic assignment for each word in corpus
topicAssignment = randi([1,nTopics],[1,nCorpus]);

% compute number of words in each doc & topic assignment 
% within each document
Nd  = zeros(1,nDocs) ;
nDocumentTopic = zeros(nDocs,nTopics);
for d = 1:nDocs
    docIndex = Ds==d;
    Nd(d) = sum(docIndex);
    for k = 1:nTopics
        nDocumentTopic(d,k) = sum(topicAssignment(docIndex)==k);
    end
end
 
% compute number of times each word in vocabulary was assigned 
% to each topic & compute number of words that was assigned 
% to each topic
nWordTopic = zeros(nVocab,nTopics);
nTopic  = zeros(1,nTopics);
for k = 1:nTopics
    topicIndex = topicAssignment==k;
    nTopic(k) = sum(topicIndex);
    for w = 1:nVocab
        nWordTopic(w,k) = sum(Ws(topicIndex)==w);
    end
end

% allocate memory for samples
samples = struct('topicAssignment',zeros(nCorpus,nSamples),...
                 'nDocumentTopic',zeros(nDocs,nTopics,nSamples),...
                 'nWordTopic',zeros(nVocab,nTopics,nSamples),...
                 'nTopic',zeros(nTopics,nSamples),...
                 'loglike',zeros(1,nSamples));
loglike = zeros(1,nSamples*nThin+nBurnin);

% ---------------------------- Start Sampling ----------------------------- 

for j = 1:(nSamples*nThin+nBurnin)
    
    % for each word in corpus sample topic assignment
    for i = 1:nCorpus
        wordId = Ws(i);
        docId  = Ds(i);
        topicId = topicAssignment(i);
        
        % remove observation j influence from all sufficient statistics
        nWordTopic(wordId,topicId) = nWordTopic(wordId,topicId) - 1;
        nDocumentTopic(docId,topicId) = nDocumentTopic(docId,topicId) - 1;
        nTopic(topicId) = nTopic(topicId) - 1;
        
        % compute p(z_{n,d} = k| Z_{-n,d}) (i.e. probability of assigning
        % topic k for wornd n in document d)
        logPLatent = log(nDocumentTopic(docId,:) + alpha) - log(alpha*nTopics + Nd(docId)-1);
        
        % compute p(W|Z) (i.e. probability of observing corpus given all
        % topic assignments)
        logPWord = log(nWordTopic(wordId,:) + gamma) - log(gamma * nVocab + nTopic);
        
        % normalise to have probability distribution
        logProbs = logPWord + logPLatent;
        probs = exp(logProbs -  logsumexp(logProbs));
        probs = probs/sum(probs);
        topicAssignment(i) = find(mnrnd(1,probs,1));
        
        % update sufficient stats using j-th variable
        topicId = topicAssignment(i);
        nWordTopic(wordId,topicId) = nWordTopic(wordId,topicId) + 1;
        nDocumentTopic(docId,topicId) = nDocumentTopic(docId,topicId) + 1;
        nTopic(topicId) = nTopic(topicId) + 1;
    end
    
    loglike(j) = joint_log_like(alpha,gamma,Nd,nTopic,nTopics,nDocs,nVocab,...
                                nDocumentTopic,nWordTopic);
    
    % accept sample after burnin & thinning
    if j > nBurnin && mod(j-nBurnin,nThin)==0
       idx = floor((j-nBurnin)/nThin);
       samples.topicAssignment(:,idx) = topicAssignment;
       samples.nDocumentTopic(:,:,idx) = nDocumentTopic;
       samples.nWordTopic(:,:,idx) = nWordTopic;
       samples.nTopic(:,idx) = nTopic;
       samples.loglike(idx) = loglike(j);
    end

end
end

