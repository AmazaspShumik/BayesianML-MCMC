% TOY EXAMPLE 

% create simple vocabulary
WO = cell(1,9);

% words related to three topics: 'Transportation','Animals',
% 'WATER'
WO{1} = 'car'; WO{2} = 'bus'; WO{3} = 'automibile';
WO{4} = 'dog'; WO{5} = 'cat'; WO{6} = 'rabbit';
WO{7} = 'sea'; WO{8} = 'ocean'; WO{9} = 'river';

% corpus size, number of topics, number of docs
nCorpus = 3000;
nTopics = 3;
nDocs = 10;

% topic distribution for generating document
pTopics = [0.8,0.1,0.1;0.1,0.8,0.1;0.1,0.1,0.8];
DS = randi([1,nDocs],[1,nCorpus]);
WS = zeros(1,nCorpus);
topics = [10,10,10,1,1,1,1,1,1;
          1,1,1,10,10,10,1,1,1;
          1,1,1,1,1,1,10,10,10];
pWordTopic = topics/36;
topicAssignment = zeros(1,nCorpus);

for j = 1:nCorpus
    y = rem(DS(j),nTopics) + 1;
    topicDist = pTopics(y,:);
    topicAssignment(j) = find(mnrnd(1,topicDist,1));
    id = find(mnrnd(1,pWordTopic(topicAssignment(j),:),1));
    WS(j) = id;
end

document_assignment  = DS;
words = WS;

% define parameters for sampling
nSamples = 2;
nBurnin  = 80;
nThin    = 2;
nTopics  = 3;
alpha    = 1; 
gamma    = 1;

% run collapsed Gibbs Sampler for LDA
[samples,loglike] = coll_gibbs_lda(words,document_assignment,nTopics,alpha,...
                              gamma,nSamples,nBurnin,nThin);

% choose top 3 words for each topic (using last sample)
topN = 3;
topWords = top_words(samples.nWordTopic(:,:,nSamples),WO,topN);

% plot joint log-likelihood
plot(loglike,'r-')
xlabel('Iterations')
ylabel('joint log-likelihood')
title('Joint log-likelihhod for collapsed Gibbs LDA')


