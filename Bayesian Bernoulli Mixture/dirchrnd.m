function sample = dirchrnd(theta)
% Simple function for sampling from dirichlet rv

y = gamrnd(theta,1);
sample = y / sum(y);
end
