close all;
clear variables;
clc;

save_run = false;

startTime = now;

if save_run
    diary output.txt
    diary on
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

for w = stftWindow
    for o = stftOverlap
        for p = stftPoints
            if o >= p
                continue
            end
            datasetSpectrum = [];
            for i = 1:size(dataset,1)
                datasetSpectrum(i,:,:) = abs(spectrogram(dataset(i,:), window(w{1}, p), o, p));
            end
            for s = stftScale
                switch s{1}
                    case 'none'
                        datasetSpectrumScale = datasetSpectrum;
                    case 'log'
                        datasetSpectrumScale = 20*log10(datasetSpectrum);
                end
                for samp = stftSamples
                    for sampo = stftSamplesOverlap
                        if samp == -1
                            samp = size(datasetSpectrumScale,2);
                        end
                        if sampo >= samp
                            continue
                        end
                        [currDatasetSpectrum, currLabels] = split_stft(datasetSpectrumScale, labels, samp, sampo);
                        trainDatasetSpectrum = currDatasetSpectrum(1:round(size(currLabels,1)*trainPercentage),:,:);
                        trainLabels = currLabels(1:round(size(currLabels,1)*trainPercentage));
                        testDatasetSpectrum = currDatasetSpectrum(round(size(currLabels,1)*trainPercentage+1):end,:,:);
                        testLabels = currLabels(round(size(currLabels,1)*trainPercentage+1):end);
                        for f = stftFeatures
                            fprintf('\nRunning stft svm for w=%s, o=%d, p=%d, s=%s, samp=%d, sampo=%d, f=%s\n',char(w{1}),o,p,s{1},samp,sampo,f{1});
                            [overall_acc, class_acc, models] = train_stft_svm(trainDatasetSpectrum, trainLabels, ...
                                testDatasetSpectrum, testLabels, f{1});
                            fprintf('Results: overall=%f, slow=%f normal=%f, high=%f\n', ...
                                overall_acc, class_acc(1), class_acc(2), class_acc(3));
                            currResult = struct('overall_acc', overall_acc, 'class_acc', class_acc, 'models', models);
                            results.(sprintf('stft_svn_w_%s_o_%d_p_%d_s_%s_samp_%d_sampo_%d_f_%s',char(w{1}),o,p,s{1},samp,sampo,f{1})) = currResult;
                        end
                    end
                end
            end
        end
    end
end

if save_run
    name = sprintf('%f.mat', startTime);
    save(name, results);
end
