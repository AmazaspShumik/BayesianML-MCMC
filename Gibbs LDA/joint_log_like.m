function [ ll ] = joint_log_like( alpha, gamma, Nd, nTopic, nTopics, nDocs,...
                                           nVocab, nDocumentTopic, nWordTopic )
% Computes joint log likelihood of observed and latent (topic assignment)
% variables i.e. p(words, latent_variable | alpha, gamma )
%
% Parameters
% ----------
% alpha: float
%        Concentration parameter for dirichlet prior of topic distribution
%
% gamma: float
%        Concentration parameter for dirichlet prior of word distribution
% 
% Nd: vector of size (1,nDocs)
%      Number of words in each document
%
% nTopic: vector of size (1,nTopics)
%      Number of words assigned to each topic
% 
% nTopics: int
%      Number of topics
% 
% nDocs: int 
%      Number of documents
%
% nVocab: int 
%      Number of unique words in corpus (vocabulary size)
%
% nDocumentTopic: matrix of size (nDocs,nTopics)
%      nDocumentTopic(d,k) - number of words assigned to topic k in
%      document d 
% 
% nWordTopic: matrix of size (nVocab,nTopics)
%      nWordTopic(v,k) - number of times word v was assigned to topic k
%
% Returns
% -------
% ll: int
%     Joint log-likelihood p(words, latent_variable | alpha, gamma )

% initialize
ll = 0;

% log of normalization constant for prior of topic distrib
ll = ll + nDocs*(gammaln(nTopics*alpha) - nTopics*gammaln(alpha));

% log of latent dist pdf without normalization constant (obtained after 
% integrating out topic distribution)
for d = 1:nDocs
 ll = ll + sum(gammaln(alpha + nDocumentTopic(d,:)));
 ll = ll - gammaln(nTopics*alpha + Nd(d));
end

% log of normalization constant for prior of word distribution
ll = ll + nTopics*gammaln(nVocab*gamma) - nVocab*gammaln(gamma);

% log p( words | latent_var), obtained after integrating out word
% distribution
for k = 1:nTopics
    ll = ll + sum( gammaln(gamma + nWordTopic(:,k)) );
    ll = ll - gammaln(nVocab*gamma + nTopic(k));
end

end
    
