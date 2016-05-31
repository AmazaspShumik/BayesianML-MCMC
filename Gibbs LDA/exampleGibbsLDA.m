load bagofwords_nips
load words_nips.mat

%number of words
alphabet_size = max(WS);

document_assignment  = DS;
words = WS;

%subset data, EDIT THIS PART ONCE YOU ARE CONFIDENT THE MODEL IS WORKING
%PROPERLY IN ORDER TO USE THE ENTIRE DATA SET
document_assignment  = document_assignment(DS <= 20);
words = words(DS <= 20);

% create sparse matrices to use in sampling
nCorpus = length(words); % corpus size
nVocab = max(words); % vocabulary size
nDocs  = max(document_assignment); % number of documents

% create sparse matrices (here we preprocess data to feed
% into )
[samples] = collapsedGibbsLDA(words,document_assignment,20);

% choose top N words for each topic





clear DS WS
