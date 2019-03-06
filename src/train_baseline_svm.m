function [ overall_acc, class_acc ] = train_baseline_svm(trainDataset, trainLabels, testDataset, testLabels)

    trainLabels_low    = trainLabels == 0;
    trainLabels_normal = trainLabels == 1;
    trainLabels_high   = trainLabels == 2;
    
    testLabels_low    = testLabels == 0;
    testLabels_normal = testLabels == 1;
    testLabels_high   = testLabels == 2;

    mdl_low    = fitcsvm(trainDataset, trainLabels_low);
    mdl_normal = fitcsvm(trainDataset, trainLabels_normal);
    mdl_high   = fitcsvm(trainDataset, trainLabels_high);
    
    [validLabels_low,    score_low]   = predict(mdl_low,     testDataset);
    [validLabels_normal, score_normal] = predict(mdl_normal, testDataset);
    [validLabels_high,   score_high]   = predict(mdl_high,   testDataset);
    
    [~, validLabels] = max([score_low(:,2) score_normal(:,2) score_high(:,2)], [], 2);
    
    overall_acc = mean(testLabels == validLabels);
    class_acc = [mean((testLabels == 0) == (validLabels == 0)); ...
                 mean((testLabels == 1) == (validLabels == 1)); ...
                 mean((testLabels == 2) == (validLabels == 2))];

end

