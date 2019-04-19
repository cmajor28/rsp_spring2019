function [Q, Lambda, R] = PCA(X)
    X = X - repmat(mean(X,2), 1, size(X,2)); % Remove mean
    R = X*X'/size(X,2); % Compute correlation matrix
    [Q, Lambda] = eig(R); % Compute eigen vectors/values
    [Lambda, ind] = sort(diag(Lambda), 'descend'); % Sort eigen values
    Q = Q(:,ind);
end
