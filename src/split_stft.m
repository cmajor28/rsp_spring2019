function [ new_dataset, new_labels ] = split_stft(dataset, labels, samples, overlap)
    
    if samples == -1 || samples == size(dataset,3)
       new_dataset = dataset;
       new_labels = labels;
    else
        new_dataset = [];
        new_labels = [];

        % Split stft dataset into subsets of each test case
        for i = 1:size(dataset,1)
            curr = 1;
            while curr + samples <= size(dataset,3)
                new_dataset(end+1,:,:) = dataset(i,:,curr:curr+samples-1);
                new_labels(end+1) = labels(i);
                curr = curr + samples - overlap;
            end
        end

        % Re-randomize labels and dataset
        idx = randperm(length(new_labels));
        new_dataset = new_dataset(idx,:,:);
        new_labels = new_labels(idx)';
    end
end

