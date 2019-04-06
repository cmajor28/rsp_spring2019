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


%This section trains the mlp model

close all
clear variables
rng(4761)
if ~exist('lown')
    load('../data/Databreath.mat');
end

data = [lown;
        Normaln;
        highn];
h1 = data;
% Limiting the test data to a 10 second interval for simplicity 
info = data(:, 1280:2560+2560);
% Limiting the data to a 5 second interval for simplicity 
%data = data(:, 1280:2560);
data = reshape(data, [], 512);
fftp = 512;
data = real(log10(fft(data, fftp, 2)));

% Data Whitening
data = data - mean(data, 2);
data = data ./ std(data, 0, 2);

labels = [ones(1, size(data, 1)/3), 2*ones(1, size(data, 1)/3), 3*ones(1, size(data, 1)/3)];
    

n = size(data, 1); %  The number of data
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

train = data(train_set, :);
test = data(test_set, :);
train_labels = labels(train_set);
test_labels = labels(test_set);

n_train = length(train_labels);
n_test = length(test_labels);



num_iterations = 2100; % Number of batches to be run 30000
eta = .000399; % Learning rate
num_hidden = 2 ; % Number of hidden layers 3
num_neurons = 700; % Number of neurons in each hidden layer 300

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
        plot(movmean(train_error, 25))
        hold on
        plot(linspace(1, num_iterations, num_iterations/test_period), test_error)
        drawnow
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
% This section tests accuracy on test data (evaluates while sliding window
% across the data) variable percent_err stores the percent error 

c = 0
counter = 0
count = 0 
error1 = zeros(1,1);
set = n_test;
batch_size = set;
window =10*256; %   256 = 1 second



selection = randperm(set, batch_size); %  select random trials from the training set
%input = data(selection, :);
%selection = (n_train+1:n_train+n_test-5);
%input = test(selection, :);
%z = zeros(1,length(input)-window);
%size = length(input(1,:));
chop = floor(length(info)/window);
mse = zeros(1, batch_size);

for h = 1:chop
    in = info(:,-window+(window*h)+1:(window*h));
    info2 = real(log10(fft(in, fftp, 2)));
    % Data Whitening
    info2 = info2 - mean(info2, 2);
    info2 = info2 ./ std(info2, 0, 2);
    test = info2(test_set, :);
    input = test(selection, :);
    for j = 1:batch_size
        c = c+1;
        %in = [input(j,init+idx+1:idx),z];
        %v_in = in * W_in;
        %v_in = input(j,-window+(window*h)+1:(window*h)) * W_in(1:window,:);
        v_in = input(j,:)*W_in;
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
        expected(test_labels(selection(j))) = 1;

        e = expected - classification;
        mse(j) = mean((e).^2);


        max_class = find(classification == max(classification))

        %calculate how many cases are misclassified and count
        if (expected(max_class) ~= 1)
            disp('.........')
            counter = counter+1
            percent_err = (counter/(0+c))*100
        elseif (length(max_class) ~=1)
            disp('.........')
            counter = counter+1
            percent_err = (counter/(0+c))*100
        end
       
    %     if mse(j)> .1
    %         disp('.........')
    %          disp(expected)
    %          disp(classification)
    %          counter = counter+1
    %     end


    end
end