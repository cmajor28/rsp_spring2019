function [ features ] = extract_pca_features( input, N )
    [Q, ~, ~] = PCA(input);
    PCs = Q(:,1:N);
    features = PCs(:);
end

