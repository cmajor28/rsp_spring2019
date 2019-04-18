function [ overall_acc, class_acc, models ] = train_stft_svm(trainDatasetStft, trainLabels, testDatasetStft, testLabels, feature)
    
    % Generate SVM features
    switch feature
        case 'none'
            trainLabels_low    = [];
            trainLabels_normal = [];
            trainLabels_high   = [];
            testLabels_low    = [];
            testLabels_normal = [];
            testLabels_high   = [];
            trainDatasetFeatures = [];
            testDatasetFeatures = [];
            idxTrain = randperm(size(trainDatasetStft,1)*size(trainDatasetStft,3));
            idxTest = randperm(size(testDatasetStft,1)*size(testDatasetStft,3));
            k=1;
            for i = 1:size(trainDatasetStft,1)
                for j = 1:size(trainDatasetStft,3)
                    trainDatasetFeatures(idxTrain(k),:) = squeeze(trainDatasetStft(i,:,j));
                    k=k+1;
                end
            end
            k=1;
            for i = 1:size(testDatasetStft,1)
                for j = 1:size(testDatasetStft,3)
                    testDatasetFeatures(idxTest(k),:) = squeeze(testDatasetStft(i,:,j));
                    k=k+1;
                end
            end
        case 'hog'
            trainLabels_low    = trainLabels == 0;
            trainLabels_normal = trainLabels == 1;
            trainLabels_high   = trainLabels == 2;
            testLabels_low    = testLabels == 0;
            testLabels_normal = testLabels == 1;
            testLabels_high   = testLabels == 2;
            trainDatasetFeatures = [];
            testDatasetFeatures = [];
            for i = 1:size(trainDatasetStft,1)
                trainDatasetFeatures(i,:) = HOG(squeeze(trainDatasetStft(i,:,:)));
            end
            for i = 1:size(testDatasetStft,1)
                testDatasetFeatures(i,:) = HOG(squeeze(testDatasetStft(i,:,:)));
            end
    end

    % Train SVMs
    mdl_low    = fitPosterior(fitcsvm(trainDatasetFeatures, trainLabels_low));
    mdl_normal = fitPosterior(fitcsvm(trainDatasetFeatures, trainLabels_normal));
    mdl_high   = fitPosterior(fitcsvm(trainDatasetFeatures, trainLabels_high));
    
    % Run SVMs
    [validLabels_low,    score_low]    = predict(mdl_low,    testDatasetFeatures);
    [validLabels_normal, score_normal] = predict(mdl_normal, testDatasetFeatures);
    [validLabels_high,   score_high]   = predict(mdl_high,   testDatasetFeatures);
    
    [~, validLabels] = max([score_low(:,2) score_normal(:,2) score_high(:,2)], [], 2);
    
    validLabels = validLabels - 1; % Stupid Matlab indexing
    
    overall_acc = mean(testLabels == validLabels);
    low_acc = get_accuracy(sum(testLabels == 0 & validLabels == 0), ...
                           sum(testLabels ~= 0 & validLabels == 0), ...
                           sum(testLabels == 0 & validLabels ~= 0), ...
                           sum(testLabels ~= 0 & validLabels ~= 0));
    normal_acc = get_accuracy(sum(testLabels == 1 & validLabels == 1), ...
                              sum(testLabels ~= 1 & validLabels == 1), ...
                              sum(testLabels == 1 & validLabels ~= 1), ...
                              sum(testLabels ~= 1 & validLabels ~= 1));
    high_acc = get_accuracy(sum(testLabels == 2 & validLabels == 2), ...
                            sum(testLabels ~= 2 & validLabels == 2), ...
                            sum(testLabels == 2 & validLabels ~= 2), ...
                            sum(testLabels ~= 2 & validLabels ~= 2));
    class_acc = [low_acc, normal_acc, high_acc];
    models = struct('low', mdl_low, 'normal', mdl_normal, 'high', mdl_high);
             
end

