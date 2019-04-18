%stftWindow = [{@rectwin} {@hamming} {@kaiser} {@taylorwin} {@chebwin}];
%stftOverlap = [0 4 8 16 32 64 128 256 512];
%stftPoints = [32 64 128 256 512 1024];
stftWindow = [{@taylorwin}];
stftOverlap = [128];
stftPoints = [256];
stftSamples = [8];
stftSamplesOverlap = [4];
stftFeatures = [{'hog'}];

useCollected = 1;
useOriginal = 0;