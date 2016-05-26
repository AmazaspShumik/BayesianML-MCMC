function [ count ] = wrongPixels( firstImg, secondImg)
% Computes number of pixels that are not the same in first and 
% second images
count = sum(sum(firstImg~=secondImg));
end

