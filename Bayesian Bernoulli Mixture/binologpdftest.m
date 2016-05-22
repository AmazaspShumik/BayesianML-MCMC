% % TEST 1
% x = zeros(100,3);
% p1 = [0.001,0.99999,0.001];
% p2 = [0.999,0.001,0.999];
% for i = 1:50
%     x(i,:) = binornd(1,p1);
%     x(50+i,:) = binornd(1,p2);
% end
% [samples,clusters,logLike] = collapsedGibbsBernoulliMixture(x,2,20,1000,2);
% [Samples,Clusters,LogLike] = vanillaGibbsBernoulliMixture(x,2,20,1000,2);


X = csvread('digits_examples.csv');
% vanilla Gibbs
[vMuSamples,vClusters,vLogLike] = vanillaGibbsBernoulliMixture(X,3,10,1000,5);
[cMuSamples,cClusters,cLogLike] = collapsedGibbsBernoulliMixture(X,3,10,1000,5);