function [ logpdf ] = binologpdf(x,p)
% BINOLOGPDF(X,probs) log pdf of Bernoulli random variable
logpdf = bsxfun(@times,x,log(p)) + bsxfun(@times,(1 - x),log(1 - p));
logpdf = sum(logpdf,2);
end

