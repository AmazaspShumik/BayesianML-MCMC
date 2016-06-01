function [ topWords  ] = top_words( nWordTopic, WO, topN )
% Finds words that have highest probability to be choosen for each topic
%
% Parameters
% ----------
% nWordTopic: matrix of size (nVocab,nTopics)
%     Number of times each word was choosen 
%
% WO: vector of size (1,nVocab)
%     Dictionary
%
% topN: int, optional (DEFAULT = 10)
%     Number of words to be selected
%
% Returns
% -------
% topWords : matrix of size (topN,nTopics)
%     Matrix of most likely words for each topic

if ~exist('topN','var')
    topN = 20;
end
topWords = cell(1,topN);
[nVocab,nTopics] = size(nWordTopic);

% it is better to use either priorty queue or partial
% sort to choose top N words but since matlab does not have
% such built-infunctions & classes we use dumb solution with 
% sorting. 
[sortedWordTopic,sortIndex] = sort(nWordTopic,'descend');
for k = 1:nTopics
    topWords{k} = WO(sortIndex(1:topN,k));
end

end

