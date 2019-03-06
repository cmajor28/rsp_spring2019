close all;
clear variables;

trainPercentage = 0.8;

load('../data/Databreath.mat');

dataset = [highn; Normaln; lown];
labels = [repmat(2, size(highn,1), 1); ...
          repmat(1, size(highn,1), 1); ...
          repmat(0, size(highn,1), 1)];

idx = randperm(size(labels,1));
dataset = dataset(idx,:);
labels = labels(idx,:);

trainDataset = dataset(1:size(labels,1)*trainPercentage,:);
trainLabels = labels(1:size(labels,1)*trainPercentage,:);
testDataset = dataset(size(labels,1)*trainPercentage+1:end,:);
testLabels = labels(size(labels,1)*trainPercentage+1:end,:);

[baseline_svm_err, baseline_svm_class_errs] = train_baseline_svm(trainDataset, trainLabels, ...
    testDataset, testLabels);

