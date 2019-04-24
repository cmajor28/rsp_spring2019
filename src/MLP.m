% Will Meaodws
% Implementation of an MLP to classify breathing micro-Doppler data. Uses a
% hyperbolic tangent activation function and gradient descent
% backpropogation 
%
%This section trains the mlp model

close all
clear variables
rng(4761)
% Choose what data set to use
% 0: Dataset we collected
% 1: Dataset from IEEE
dataset_select = 0;
if(dataset_select)
    load('../data/Databreath.mat');
    data = [lown;
        Normaln;
        highn];
else
    load('../data/micro-Doppler_collection.mat');
    data = [slow;
        normal;
        high;];
end


h1 = data;
% Limiting the test data to a 10 second interval for simplicity 
info = data(:, 1280:2560+2560);

sample_length = 256; % How many samples for each input @ 256 samples/second
data = reshape(data, [], sample_length);
fftp = 256; % The length of the FFT
data = fftshift(real(log10(fft(data, fftp, 2))));

% Data Whitening
data = data - mean(data(:));
data = data ./ std(data(:));


labels = [ones(1, size(data, 1)/3), 2*ones(1, size(data, 1)/3), 3*ones(1, size(data, 1)/3)];
    

n = size(data, 1); %  The number of data
train_ratio = .85; %  80% of data used for training 20% for testing
random_set = randperm(n);
train_set = random_set(1: floor(train_ratio * n));
test_set = random_set(floor(train_ratio * n)+1 : end);

train = data(train_set, :);
test = data(test_set, :);
train_labels = labels(train_set);
test_labels = labels(test_set);

n_train = length(train_labels);
n_test = length(test_labels);



num_iterations = 2000; % Number of batches to be run 30000
eta = .001; % Learning rate
num_hidden = 2 ; % Number of hidden layers 3
num_neurons = 150; % Number of neurons in each hidden layer 300
%  Feedforward
batch_size = 20;
num_classes = 3;
W_in =  rand(size(data, 2), num_neurons) - .5;
W_h = rand(num_neurons , num_neurons, num_hidden -1) - .5;
W_out = rand(num_neurons, num_classes) - .5;
dW_in = zeros(size(data, 2), num_neurons);
dW_h = zeros(num_neurons, num_neurons, num_hidden -1);
dW_out = zeros(num_neurons, num_classes);
test_period = 50;
train_error = zeros(1,num_iterations); % Vector to keep track of error for plotting
test_error = zeros(1, floor(num_iterations/test_period));
figure
for i = 1:num_iterations
    selection = randperm(n_train, batch_size); %  select random trials from the training set
    input = train(selection, :);
    
    [y_out, dW_in, dW_h, dW_out, mse] =  MLP_prop(input, W_in, W_h, W_out, train_labels(selection), eta);
    
    W_in = W_in + dW_in;
    W_h = W_h + dW_h;
    W_out = W_out + dW_out;
    
    train_error(i) = mean(mse);
    if ~mod(i, test_period)
        [y_out, dW_in, dW_h, dW_out, mse] =  MLP_prop(test, W_in, W_h, W_out, test_labels, eta);
        test_error(i/test_period) = mean(mse);
        plot(movmean(train_error, 25), 'DisplayName', 'Training Error')
        hold on
        plot(linspace(1, num_iterations, num_iterations/test_period), test_error, 'DisplayName', 'Test Error')
        drawnow
        legend
        xlabel('Iterations')
        ylabel('Mean squared error')
        title('Training Accuracy')
        hold off
    end
end

save('weights.mat', 'W_in', 'W_h','W_out')
%%
%This section tests accuracy on training data
%variable trainpercent_err stores the percent error

[y_out, dW_in, dW_h, dW_out, mse] =  MLP_prop(train, W_in, W_h, W_out, train_labels, eta);
[value, max_class] = max(y_out);
train_correct = sum(max_class == train_labels);
sprintf("Correctly classified (%i/%i, %.2f) from the training set)", train_correct, n_train, (train_correct/n_train))

[y_out, dW_in, dW_h, dW_out, mse] =  MLP_prop(test, W_in, W_h, W_out, test_labels, eta);
[value, max_class] = max(y_out);
test_correct = sum(max_class == test_labels);
sprintf("Correctly classified (%i/%i, %.2f) from the testing set)", test_correct, n_test, (test_correct/n_test))

%%
% Perform classification on the opposite dataset than the network was
% trained on.
if (0)
    if dataset_select
        load('../data/micro-Doppler_collection.mat');
        data_other = [slow;
            normal;
            high;];
    else
        load('../data/Databreath.mat');
        data_other = [lown;
            Normaln;
            highn];
    end
    data_other = reshape(data_other, [], sample_length);
   	data_other = real(log10(fft(data_other, fftp, 2)));
    % Data Whitening
    data_other = data_other - mean(data_other(:));
    data_other = data_other ./ std(data_other(:));
    
    n_data_other = size(data_other, 1);
    labels = [ones(1, size(data_other, 1)/3), 2*ones(1, size(data_other, 1)/3), 3*ones(1, size(data_other, 1)/3)];
    load('weights.mat');
    [y_out, dW_in, dW_h, dW_out, mse] =  MLP_prop(data_other, W_in, W_h, W_out, labels, eta);
    [value, max_class] = max(y_out);
    num_correct = sum(max_class == labels);
    sprintf("Correctly classified (%i/%i, %.2f) from the opposite dataset)", num_correct, n_data_other, (num_correct/n_data_other))
end
