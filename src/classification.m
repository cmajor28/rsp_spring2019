close all;
clear variables;
clc;

warning('on','all');

save_run = true;

startTime = now;

if save_run
    warning('off','all');
    diary off;
    diary output.txt;
    diary on;
end

fprintf('\n\nStarting Run: %f\n', startTime);

run params;
trainPercentage = 0.8;

dataset = [];
labels = [];

if useCollected
   load('../data/Subject1_slow.mat');
   dataset = [dataset; d.'];
   labels = [labels; 0];
   load('../data/Subject2_slow.mat');
   dataset = [dataset; d.'];
   labels = [labels; 0];
   load('../data/Subject3_slow.mat');
   dataset = [dataset; d.']; 
   labels = [labels; 0];
   load('../data/Subject4_slow.mat');
   dataset = [dataset; d.']; 
   labels = [labels; 0];
   load('../data/Subject1_normal.mat');
   dataset = [dataset; d.']; 
   labels = [labels; 1];
   load('../data/Subject2_normal.mat');
   dataset = [dataset; d.']; 
   labels = [labels; 1];
   load('../data/Subject3_normal.mat');
   dataset = [dataset; d.']; 
   labels = [labels; 1];
   load('../data/Subject4_normal.mat');
   dataset = [dataset; d.']; 
   labels = [labels; 1];
   load('../data/Subject4_high.mat');
   dataset = [dataset; d.']; 
   labels = [labels; 2];
   load('../data/Subject4_high.mat');
   dataset = [dataset; d.']; 
   labels = [labels; 2];
   load('../data/Subject4_high.mat');
   dataset = [dataset; d.']; 
   labels = [labels; 2];
   load('../data/Subject4_high.mat');
   dataset = [dataset; d.'];
   labels = [labels; 2];
   dataset = dataset / max(dataset(:));
elseif useOriginal
    load('../data/Databreath.mat');
    dataset = [dataset; highn; Normaln; lown];
    labels = [labels; ...
              repmat(2, size(highn,1), 1); ...
              repmat(1, size(highn,1), 1); ...
              repmat(0, size(highn,1), 1)];
end

idx = randperm(size(labels,1));
dataset = dataset(idx,:);
labels = labels(idx,:);

results = struct();

for win = stftWindow
    for wsize = stftWindowSize
        if wsize == -1
            wsize = size(dataset,2);
        end
        for woverlap = stftWindowOverlap
            if woverlap >= wsize
                continue
            end
            for points = stftPoints
                datasetSpectrum = [];
                for i = 1:size(dataset,1)
                    datasetSpectrum(i,:,:) = abs(spectrogram(dataset(i,:), window(win{1}, wsize), woverlap, points));
                end
                for scale = stftScale
                    switch scale{1}
                        case 'none'
                            datasetSpectrumScale = datasetSpectrum;
                        case 'log'
                            datasetSpectrumScale = 20*log10(datasetSpectrum);
                    end
                    for samp = stftSamples
                        for sampoverlap = stftSamplesOverlap
                            if samp == -1
                                samp = size(datasetSpectrumScale,3);
                            end
                            if sampoverlap >= samp
                                continue
                            end
                            [currDatasetSpectrum, currLabels] = split_stft(datasetSpectrumScale, labels, samp, sampoverlap);
                            
                            if isempty(currDatasetSpectrum)
                                continue;
                            end
                            
                            trainDatasetSpectrum = currDatasetSpectrum(1:round(size(currLabels,1)*trainPercentage),:,:);
                            trainLabels = currLabels(1:round(size(currLabels,1)*trainPercentage));
                            testDatasetSpectrum = currDatasetSpectrum(round(size(currLabels,1)*trainPercentage+1):end,:,:);
                            testLabels = currLabels(round(size(currLabels,1)*trainPercentage+1):end);
                            
                            for feature = stftFeatures
                                fprintf('\nRunning stft svm for window=%s, windowSize=%d windowOverlap=%d, stftPoints=%d, scale=%s, samples=%d, sampleOverlap=%d, feature=%s\n',...
                                    char(win{1}), wsize, woverlap, points, scale{1}, samp, sampoverlap, feature{1});
                                [overall_acc, class_acc, models] = train_stft_svm(trainDatasetSpectrum, trainLabels, ...
                                    testDatasetSpectrum, testLabels, feature{1});
                                fprintf('Results: overall=%f, slow=%f normal=%f, high=%f\n', ...
                                    overall_acc, class_acc(1), class_acc(2), class_acc(3));
                                currResult = struct('overall_acc', overall_acc, 'class_acc', class_acc);
                                name = sprintf('stft_svn_%s_%d_%d_%d_%s_%d_%d_%s', ...
                                    char(win{1}),wsize, woverlap, points, scale{1}, samp, sampoverlap, feature{1});
                                results.(name) = currResult;
                            end
                        end
                    end
                end
            end
        end
    end
end

if save_run
    name = sprintf('%d.mat', round(startTime*10^6));
    save(name, 'results');
end
