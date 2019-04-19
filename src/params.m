%stftWindow = [{@rectwin} {@hamming} {@kaiser} {@taylorwin} {@chebwin}];
%stftOverlap = [0 4 8 16 32 64 128 256 512];
%stftPoints = [32 64 128 256 512 1024];
stftWindow = [{@taylorwin}];
stftOverlap = [128];
stftPoints = [256];
stftSamples = [-1];
stftSamplesOverlap = [32];
stftFeatures = [{'none'}, {'pca4'}, {'pca8'}, {'pca16'}, {'statistical'}, {'hog'}];
stftScale = [{'log'}, {'none'}];

useCollected = 0;
useOriginal = 1;