function [ features ] = extract_statistical_features( input )
    features = [mean(input(:)), var(input(:))];
end

