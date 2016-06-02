load bagofwords_nips
load words_nips.mat
load titles_nips.mat

%number of words
alphabet_size = max(WS);

document_assignment  = DS;
words = WS;

%subset data, EDIT THIS PART ONCE YOU ARE CONFIDENT THE MODEL IS WORKING
%PROPERLY IN ORDER TO USE THE ENTIRE DATA SET
document_assignment  = document_assignment(DS <= 100);
words = words(DS <= 100);


% define parameters for sampling
nSamples = 2;
nBurnin  = 20;
nThin    = 2;
nTopics  = 20;
alpha    = 1; 
gamma    = 1;

% run collapsed Gibbs Sampler for LDA
[samples,loglike] = coll_gibbs_lda(words,document_assignment,nTopics,alpha,...
                              gamma,nSamples,nBurnin,nThin);
                          
% choose top 5 words for each topic (using last sample)
topN = 5;
topWords = top_words(samples.nWordTopic(:,:,nSamples),WO,topN);

% plot joint log-likelihood
plot(loglike,'b-')
xlabel('Iterations')
ylabel('joint log-likelihood')
title('Joint log-likelihhod for collapsed Gibbs LDA')
