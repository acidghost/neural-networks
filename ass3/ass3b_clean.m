clear all; close all; clc;
load('pics.mat');

fold_size = 10;
[ n_examples, nin ] = size(pics);
nout = 1;
cycles = 100;
alpha = 0.01;
% hidden_nodes = [1, 2, 3, 4, 5, 6, 7, 15, 25, 50, 100, 200, 1000, nin];
hidden_nodes = [ 2, 3, 4, 5, 7, 10, 15, 25 ];
options = zeros(18, 1);
options(1) = 1;
options(14) = cycles;
options(17) = alpha;

% gaborArray = gaborFilterBank(5, 8, 39, 39);

% Preprocess data
% x = pics - mean(mean(pics));
% t = classGlass - mean(classGlass);
x = pics;
% Gabor feature vector length: (m*n*u*v)/(d1*d2)
% for i = 1:size(pics, 1);
%     x(i, :) = gaborFeatures(pics(i, :), gaborArray, 8, 8)';
% end
% nin = length(x(1, :));
t = classGlass;
t(t == 0) = -1;

% Calculate indices for K-fold cross validation
fold_indices = ass3a_crossvalid(fold_size, n_examples);

% Shuffle data for cross validation
permutation = randperm(n_examples);
x = x(permutation, :);
t = t(permutation);

confmat_by_hidden_nodes.nodes = hidden_nodes;
confmat_by_hidden_nodes.matrices = zeros(2, 2, length(hidden_nodes));
for i = 1:length(hidden_nodes);
    nhidden = hidden_nodes(i);
    % Use K-fold cross validation
    confusionmat = zeros(2, 2, fold_size);
    for j = 1:fold_size;
        fold_index = fold_indices(j, :);
        % Use j-th block for testing
        x_test = x(fold_index, :);
        t_test = t(fold_index)';
        % Use remaining for training
        train_indices = setdiff(1:n_examples, fold_index);
        x_train = x(train_indices, :);
        t_train = t(train_indices)';
        
        % Init network
        net = mlp(nin, nhidden, nout, 'linear');
        for k = 1:cycles;
            % Feed-forward the inputs through the network
            [Y, Z, A] = mlpfwd(net, x_train);
            
            Y(Y >= 0) = 1;
            Y(Y < 0) = -1;

            % Back-propagate the error
            G = mlpbkp(net, x_train, Z, Y - t_train);

            % Update weights in network
            old_weights = netpak(net);
            weights = old_weights - alpha * G;
            net = netunpak(net, weights);
        end
        
        % Do backprop (graddesc)
        % net = netopt(net, options, x_train, t_train, 'graddesc');
        
        Y = mlpfwd(net, x)';
        Y(Y >= 0) = 1;
        Y(Y < 0) = -1;
        true_positive = sum(Y == 1 & t == 1);
        false_positive = sum(Y == 1 & t == -1);
        true_negative = sum(Y == -1 & t == 1);
        false_negative = sum(Y == -1 & t == -1);
        confusionmat(:, :, j) = [ true_positive, true_negative; false_positive, false_negative ];
    end
    
    confmat_by_hidden_nodes.matrices(:, :, i) = [ mean(confusionmat(1, 1, :)), mean(confusionmat(1, 2, :)); mean(confusionmat(2, 1, :)), mean(confusionmat(2, 2, :)) ]
end

disp(confmat_by_hidden_nodes)
