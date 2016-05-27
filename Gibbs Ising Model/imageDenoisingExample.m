% This script illustrates image denoisning using Gibbs Sample 
% 
% (image in this example is taken from Statistical Machine Learning 
% course taught by Frank Wood in Spring 2012 at Columbia University)

% load data 
data = load('data');
realImg  = data.img;
noisyImg = data.noisy_img;

% parameters of Gibbs Sample
couplingStrength = -1;
externalStrength = -1;
nBurnin          = 200;
nSamples         = 100;
nThin            = 3;

% samples after burnin and thinning
samples = gibbsIsingModel(noisyImg,couplingStrength,externalStrength,...
                          nSamples,nBurnin,nThin);                      
                      
% vizualise noisy image
figure(1)
imshow(noisyImg)
title('Noisy Image')
                      
% vizualise last sample
figure(2)
imshow(samples(:,:,nSamples))
title('Denoised Image, sample from posterior')

% vizualise picture without noise
figure(3)
imshow(realImg)
title('Real Image, without noise')

% compute proportion of 'wrong' pixels in each sample
err = zeros(1,nSamples);
for i = 1:nSamples
    err(i) = wrongPixels(samples(:,:,i),realImg);
end
[R,C] = size(realImg);
nPixels = R*C;
err = err / nPixels;

% plot histogram of number of wrong pixels
figure(4)
hist(err)
title('Proportion of wrong pixels in samples. Initial noise = 0.0973')




