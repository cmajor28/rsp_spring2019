function [y_all, dW_in, dW_h, dW_out, mse] = MLP_prop(input, W_in, W_h, W_out, labels, eta)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

dW_in = zeros(size(W_in));
dW_h = zeros(size(W_h));
dW_out = zeros(size(W_out));
batch_size = size(input, 1);
mse = zeros(1, batch_size);
y_all = zeros(size(W_out, 2) ,batch_size);
num_hidden = size(dW_h, 3) +1;
num_neurons = size(dW_h, 1);

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
        y_all(:, j) = y_out;
        
        
        classification = y_out;  %3 different prediciton probabilities
        expected = -1 * ones(1, 3);
        expected(labels(j)) = 1;
        
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
    
end