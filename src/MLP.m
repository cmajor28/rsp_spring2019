% Will Meaodws
% Implementation of an MLP to classify breathing micro-Doppler data. Uses a
% hyperbolic tangent activation function and gradient descent
% backpropogation 
%
% TODO:
%   Tweak values and probably train longer to reduce training error
%   Begin evaluating using test set
%   Stretch goal: plot realtime training and validation results
%   Variable learning rate or change activation function to leaky relu

close all
clear variables
rng('shuffle')
if ~exist('lown')
    load('../data/Databreath.mat');
end

data = [lown';
        Normaln';
        highn'];
h1 = data;
% Segment the data into 5 second intervals
data = reshape(data, 1280, []);
data = real(log10(fft(data, 2048)));

% Data Whitening
% The dataset is already 0 mean
%data = data - mean(data, 2);
data = data ./ std(data, 0, 'all');

labels = [ones(1, size(data, 2)/3), 2*ones(1, size(data, 2)/3), 3*ones(1, size(data, 2)/3)];
    

n = length(labels); %  The number of data
train_ratio = .99; %  80% of data used for training 20% for testing
random_set = randperm(n);
train_set = random_set(1: floor(train_ratio * n));
test_set = random_set(floor(train_ratio * n)+1 : end);
% set1 = 1:75;
% set2 = 76:150;
% set3 = 151:225;
% train_set = set1(1: floor(train_ratio * 75));
% train_set = [train_set set2(1: floor(train_ratio * 75))];
% train_set = [train_set set3(1: floor(train_ratio * 75))];
% 
% test_set = set1(floor(train_ratio * 75)+1 : end);
% test_set = [test_set set2(floor(train_ratio * 75)+1 : end)];
% test_set = [test_set set3(floor(train_ratio * 75)+1 : end)];

train = data(:, train_set);
test = data(:, test_set);
train_labels = labels(train_set);
test_labels = labels(test_set);

n_train = length(train_labels);
n_test = length(test_labels);



num_iterations = 1250;%2100; % Number of batches to be run 30000
eta = .0000096; % Learning rate
num_hidden = 3 ; % Number of hidden layers 3
num_neurons = 300; % Number of neurons in each hidden layer 300

%  Feedforward
batch_size = 20;
num_classes = 3;
W_in =  rand(size(data, 1), num_neurons) - .5;
W_h = rand(num_neurons , num_neurons, num_hidden -1) - .5;
W_out = rand(num_neurons, ceil(log2(num_classes))) - .5;

test_period = 50; % How often should the data be validated
train_error = zeros(1,num_iterations); % Vector to keep track of error for plotting
test_error = zeros(1, floor(num_iterations/test_period));
for i = 1:num_iterations
    selection = randperm(n_train, batch_size); %  select random trials from the training set
    input = train(:, selection);
    
    
    
    % Long function that handles MLP forward and backward propogation
    [y_out, dW_in, dW_h, dW_out, mse] =  MLP_prop(input, W_in, W_h, W_out, train_labels(selection), eta);
    
    W_in = W_in + dW_in;
    W_h = W_h + dW_h;
    W_out = W_out + dW_out;
    
    train_error(i) = mean(mse);
    if ~mod(i, test_period)
        for j = 1:size(test, 2)
            [y_out, dW_in, dW_h, dW_out, mse] =  MLP_prop(test, W_in, W_h, W_out, test_labels, eta);
            test_error(i/test_period) = mean(mse);
        end
    end
end

figure;
plot(movmean(train_error, 5))
hold on
title('Training Error')
plot(linspace(1, num_iterations, num_iterations/test_period), test_error)
save('weights.mat', 'W_in', 'W_h','W_out')

