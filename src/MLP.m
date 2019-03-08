% Will Meaodws
% Implementation of an MLP to classify breathing micro-Doppler data. Uses a
% hyperbolic tangent activation function and gradient descent
% backpropogation 
%
% TODO:
%   Tweak values and probably train longer to reduce training error
%   Begin evaluating using test set
%   Stretch goal: plot realtime training and validation results

close all
clear variables

if ~exist('lown')
    load('../data/Databreath.mat');
end

data = [lown;
        Normaln;
        highn];
% Limiting the data to a 5 second interval for simplicity
data = data(:, 1280:2560);
data = real(log(fft(data, 2048, 2)));

% Data Whitening
data = data - mean(data, 2);
data = data ./ std(data, 0, 2);

labels = [ones(size(lown, 1), 1);
        2*ones(size(Normaln, 1), 1);
        3*ones(size(highn, 1), 1)];
    

n = size(data, 1); %  The number of data
train_ratio = .8; %  80% of data used for training 20% for testing
random_set = randperm(n);
train_set = random_set(1: floor(train_ratio * n));
test_set = random_set(floor(train_ratio * n)+1 : end);
train = data(train_set, :);
test = data(test_set, :);
train_labels = labels(train_set);
test_labels = labels(test_set);

n_train = length(train_labels);
n_test = length(test_labels);



num_iterations = 1000; % Number of batches to be run
eta = .000001; % Learning rate
num_hidden = 5; % Number of hidden layers
num_neurons = 200; % Number of neurons in each hidden layer

%  Feedforward
batch_size = 5;
num_classes = 3;
W_in =  rand(size(data, 2), num_neurons) - .5;
W_h = rand(num_neurons , num_neurons, num_hidden -1) - .5;
W_out = rand(num_neurons, num_classes) - .5;
dW_in = zeros(size(data, 2), num_neurons);
dW_h = zeros(num_neurons, num_neurons, num_hidden -1);
dW_out = zeros(num_neurons, num_classes);
error = zeros(1,num_iterations); % Vector to keep track of error for plotting
for i = 1:num_iterations
    selection = randperm(n_train, batch_size); %  select random trials from the training set
    input = data(selection, :);
    mse = zeros(1, batch_size);
    for j = 1:batch_size
        v_in = input(j,:) * W_in;
        y_in = tanh(v_in);
        v_h = zeros(num_hidden -1, num_neurons );
        y_h = zeros(num_hidden - 1, num_neurons);
        y = y_in;
        for k = 1:(num_hidden - 1)
            v_h(k, :) = y * W_h(:,:,k);
            y = tanh(v_h(k, :));
            y_h(k, :) = y;
        end
        
        v_out = y * W_out;
        y_out = tanh(v_out);
        
        classification = y_out;
        expected = -1 * ones(1, 3);
        expected(train_labels(selection(j))) = 1;
        
        e = expected - classification;
        mse(j) = mean((e).^2);
        
        % Backpropogation
        
        
        %start with the output layer
        grad = 1 - tanh(v_out).^2;
        delta = e .* grad;
        dW_out = dW_out + eta * y_h(end, :).' * delta;
        
        % hidden layers
        W = W_out;
        y = [y_in;
            y_h];
        for k = (num_hidden - 1) : -1 : 1
            grad = 1 - tanh(v_h(k, :)).^2;
            delta = grad .* (delta * W.');
            dW_h(:, :, k) = dW_h(:,:,k) + eta * (y(k,:).' * delta);
            W = W_h(:,:,k);
        end
        
        % input layer
        grad = 1 - tanh(v_in).^2;
        delta = grad .* (delta * W_h(:, :, 1).');
        dW_in = dW_in + eta * (input(j,:).' * delta);
        
        
    end
    W_in = W_in + dW_in;
    W_h = W_h + dW_h;
    W_out = W_out + dW_out;
    
    error(i) = mean(mse);
end

figure;
plot(error)
title('Training Error')
