function [ overall_acc, class_acc, models ] = train_stft_svm(trainDatasetStft, trainLabels, testDatasetStft, testLabels, feature)
    
    trainLabels_low    = trainLabels == 0;
    trainLabels_normal = trainLabels == 1;
    trainLabels_high   = trainLabels == 2;
    testLabels_low    = testLabels == 0;
    testLabels_normal = testLabels == 1;
    testLabels_high   = testLabels == 2;
    trainDatasetFeatures = [];
    testDatasetFeatures = [];    

    % Generate SVM features
    switch feature
    case 'none'
        for i = 1:size(trainDatasetStft,1)
            data = squeeze(trainDatasetStft(i,:,:));
            trainDatasetFeatures(i,:) = data(:);
        end
        for i = 1:size(testDatasetStft,1)
            data = squeeze(testDatasetStft(i,:,:));
            testDatasetFeatures(i,:) = data(:);
        end      
    case 'hog'
        for i = 1:size(trainDatasetStft,1)
            trainDatasetFeatures(i,:) = extract_hog_features(squeeze(trainDatasetStft(i,:,:)));
        end
        for i = 1:size(testDatasetStft,1)
            testDatasetFeatures(i,:) = extract_hog_features(squeeze(testDatasetStft(i,:,:)));
        end
    case 'statistical'
        for i = 1:size(trainDatasetStft,1)
            trainDatasetFeatures(i,:) = extract_statistical_features(squeeze(trainDatasetStft(i,:,:)));
        end
        for i = 1:size(testDatasetStft,1)
            testDatasetFeatures(i,:) = extract_statistical_features(squeeze(testDatasetStft(i,:,:)));
        end
    case 'pca4'
        for i = 1:size(trainDatasetStft,1)
            trainDatasetFeatures(i,:) = extract_pca_features(squeeze(trainDatasetStft(i,:,:)), 4);
        end
        for i = 1:size(testDatasetStft,1)
            testDatasetFeatures(i,:) = extract_pca_features(squeeze(testDatasetStft(i,:,:)), 4);
        end
    case 'pca8'
        for i = 1:size(trainDatasetStft,1)
            trainDatasetFeatures(i,:) = extract_pca_features(squeeze(trainDatasetStft(i,:,:)), 8);
        end
        for i = 1:size(testDatasetStft,1)
            testDatasetFeatures(i,:) = extract_pca_features(squeeze(testDatasetStft(i,:,:)), 8);
        end
    case 'pca16'
        for i = 1:size(trainDatasetStft,1)
            trainDatasetFeatures(i,:) = extract_pca_features(squeeze(trainDatasetStft(i,:,:)), 16);
        end
        for i = 1:size(testDatasetStft,1)
            testDatasetFeatures(i,:) = extract_pca_features(squeeze(testDatasetStft(i,:,:)), 16);
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

