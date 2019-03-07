close all;
clear variables;

run params;
trainPercentage = 0.8;

load('../data/Databreath.mat');

dataset = [highn; Normaln; lown];
labels = [repmat(2, size(highn,1), 1); ...
          repmat(1, size(highn,1), 1); ...
          repmat(0, size(highn,1), 1)];

idx = randperm(size(labels,1));
dataset = dataset(idx,:);
labels = labels(idx,:);

results = struct();

trainDataset = dataset(1:size(labels,1)*trainPercentage,:);
trainLabels = labels(1:size(labels,1)*trainPercentage,:);
testDataset = dataset(size(labels,1)*trainPercentage+1:end,:);
testLabels = labels(size(labels,1)*trainPercentage+1:end,:);

for w = stftWindow
    for o = stftOverlap
        for p = stftPoints
            if o >= p
                continue
            end
            trainDatasetSpectrum = [];
            testDatasetSpectrum = [];
            for i = 1:size(trainDataset,1)
                trainDatasetSpectrum(i,:,:) = abs(spectrogram(trainDataset(i,:), window(w{1}, p), o, p));
            end
            for i = 1:size(testDataset,1)
                testDatasetSpectrum(i,:,:) = abs(spectrogram(testDataset(i,:), window(w{1}, p), o, p));
            end
            for f = stftFeatures
                fprintf('Running stft svm for w=%s, o=%d, p=%d, f=%s\n',char(w{1}),o,p,f{1});
                [overall_err, class_err] = train_stft_svm(trainDatasetSpectrum, trainLabels, ...
                    testDatasetSpectrum, testLabels, f{1});
                fprintf('Results: overall=%f, slow=%f normal=%f, high=%f\n', ...
                    overall_err, class_err(1), class_err(2), class_err(3));
                results = struct('overall_error', overall_err, 'class_error', class_err);
                results.(sprintf('stft_svn_w_%s_o_%d_p_%d_f_%s',char(w{1}),o,p,f{1})) = results;
            end
        end
    end
end

results

% [baseline_svm_err, baseline_svm_class_errs] = train_baseline_svm(trainDataset, trainLabels, ...
%     testDataset, testLabels);

