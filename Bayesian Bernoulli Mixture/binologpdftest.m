% TEST 1
x = zeros(100,3);
p1 = [0.001,0.99999,0.001];
p2 = [0.999,0.001,0.999];
for i = 1:50
    x(i,:) = binornd(1,p1);
    x(50+i,:) = binornd(1,p2);
end
[samples,clusters] = collapsedGibbsBernoulliMixture(x,2,20,100,2);

% % Digit Test
% %D = csvread('train.csv',1);
% % select 6,3,1
% X   = zeros(30,784);
% % zeros
% idx0       = find(D(:,1)==6);
% X(1:100,:) = D(idx0(1:100),2:785);
% % ones
% idx1       = find(D(:,1)==1);
% X(101:200,:) = D(idx1(1:100),2:785);
% % threes
% idx3       = find(D(:,1)==3);
% X(201:300,:) = D(idx3(1:100),2:785);
% X( X > 0 ) = 1;
% 
% samples = vanillaGibbsBernoulliMixture(X,3,100,2000,50);