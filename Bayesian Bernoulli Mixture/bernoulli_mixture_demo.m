X = zeros(1000,3);
X(1:330,1) = 1;
X(330:700,2) = 1;
X(700:1000,3)  =1;
k=3;
maxIter = 8;
convThresh = 1e-2;
bmm        = BernoulliMixture(k,maxIter,convThresh);
bmm.fit(X)
plot(bmm.logLike);