function [ overall_acc, class_acc ] = train_stft_svm(trainDatasetStft, trainLabels, testDatasetStft, testLabels, feature)

    trainLabels_low    = trainLabels == 0;
    trainLabels_normal = trainLabels == 1;
    trainLabels_high   = trainLabels == 2;
    
    testLabels_low    = testLabels == 0;
    testLabels_normal = testLabels == 1;
    testLabels_high   = testLabels == 2;
    
    trainDatasetFeatures = [];
    testDatasetFeatures = [];
    
    for i = 1:size(trainDatasetStft,1)
        switch feature
            case 'hog'
                trainDatasetFeatures(i,:) = HOG(squeeze(trainDatasetStft(i,:,:)));
            otherwise
                testDatasetFeatures(i,:) = 0;
        end
    end
    for i = 1:size(testDatasetStft,1)
        switch feature
            case 'hog'
                testDatasetFeatures(i,:) = HOG(squeeze(testDatasetStft(i,:,:)));
            otherwise
                testDatasetFeatures(i,:) = 0;
        end
    end

    mdl_low    = fitcsvm(trainDatasetFeatures, trainLabels_low);
    mdl_normal = fitcsvm(trainDatasetFeatures, trainLabels_normal);
    mdl_high   = fitcsvm(trainDatasetFeatures, trainLabels_high);
    
    [validLabels_low,    score_low]    = predict(mdl_low,    testDatasetFeatures);
    [validLabels_normal, score_normal] = predict(mdl_normal, testDatasetFeatures);
    [validLabels_high,   score_high]   = predict(mdl_high,   testDatasetFeatures);
    
    [~, validLabels] = max([score_low(:,2) score_normal(:,2) score_high(:,2)], [], 2);
    
    validLabels = validLabels - 1; % Stupid Matlab indexing
    
    overall_acc = mean(testLabels == validLabels);
    class_acc = [mean((testLabels == 0) == (validLabels == 0)); ...
                 mean((testLabels == 1) == (validLabels == 1)); ...
                 mean((testLabels == 2) == (validLabels == 2))];

end

