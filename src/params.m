stftWindow = [{@hamming} {@rectwin} {@taylorwin}];
stftWindowSize = [16, 64, 256];
stftWindowOverlap = [4, 16, 64];
stftPoints = [32, 128, 512];
stftSamples = [32, 64];
stftSamplesOverlap = [8, 16];
stftFeatures = [{'none'}, {'pca16'}, {'statistical'}, {'hog'}];
stftScale = [{'log'}];

% stftWindow = [{@hamming}];
% stftWindowSize = [32];
% stftWindowOverlap = [8];
% stftPoints = [32];
% stftSamples = [128];
% stftSamplesOverlap = [32];
% stftFeatures = [{'hog'}];
% stftScale = [{'log'}];

useCollected = 1;
useOriginal = 0;