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
rng('shuffle')
if ~exist('lown')
    load('Databreath.mat');
end

data = [lown;
        Normaln;
        highn];
h1 = data;
% Limiting the test data to a 10 second interval for simplicity 
info = data(:, 1280:2560+2560);
% Limiting the data to a 5 second interval for simplicity 
data = data(:, 1280:2560);

fftp = 1064
data = real(log10(fft(data, fftp, 2)));

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
eta = .0000399; % Learning rate
num_hidden = 3 ; % Number of hidden layers 3
num_neurons = 300; % Number of neurons in each hidden layer 300

%  Feedforward
batch_size = 20;
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
    input = train(selection, :);
    mse = zeros(1, batch_size);
    for j = 1:batch_size    %one batch at a time for classification
        v_in = input(j,:) * W_in;
        y_in = tanh(v_in);
        %y_in = max(0,v_in);
        
        v_h = zeros(num_hidden -1, num_neurons );
        y_h = zeros(num_hidden - 1, num_neurons);
        y = y_in;
        for k = 1:(num_hidden - 1)
            v_h(k, :) = y * W_h(:,:,k);
            y = tanh(v_h(k, :));
            %y = max(0,v_h(k, :));
            
            y_h(k, :) = y;
        end
        
        v_out = y * W_out;
        y_out = tanh(v_out);
        
        
        classification = y_out;  %3 different prediciton probabilities
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

save('weights.mat', 'W_in', 'W_h','W_out')
%%
%This section tests accuracy on training data
%variable trainpercent_err stores the percent error

train_counter = 0
error1 = zeros(1,1);
set = n_train
batch_size = set;

selection = randperm(set, batch_size); %  select random trials from the training set
%input = data(selection, :);
%selection = (n_train+1:n_train+n_test-5);

input = train(selection, :);
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


    max_class = find(classification == max(classification))

    %calculate how many cases are misclassified and count
    if (expected(max_class) ~= 1)
        disp('.........')
        train_counter = train_counter+1
        trainpercent_err = (train_counter/j)*100
    elseif (length(max_class) ~=1)
        disp('.........')
        train_counter = train_counter+1
        trainpercent_err = (train_counter/j)*100
    end
%     if mse(j)> .1
%         disp('.........')
%          disp(expected)
%          disp(classification)
%          counter = counter+1
%     end

end
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
