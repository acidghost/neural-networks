clear; close all; clc;
load('pics.mat');

do_gabor = true;
fold_size = 10;
[ n_examples, nin ] = size(pics);
nout = 1;
cycles = 50;
alpha = 0.01;
hidden_nodes = [1, 2, 3, 4, 5, 7, 10, 15, 25, 50, 100, 200, (n_examples - (n_examples/fold_size))];
% hidden_nodes = [ 2, 3, 4, 5, 7, 10, 15, 25 ];
options = zeros(18, 1);
options(1) = 1;
options(14) = cycles;
options(17) = alpha;

if do_gabor
    gaborArray = gaborFilterBank(5, 8, 39, 39);
    % Gabor feature vector
    img1 = gaborFeatures(pics(1, :), gaborArray, 8, 8)';
    x = zeros(n_examples, length(img1));
    x(1, :) = img1;
    for i = 2:size(pics, 1);
        disp(['Doing Gabor image ', num2str(i)])
        x(i, :) = gaborFeatures(pics(i, :), gaborArray, 8, 8)';
    end
else
    x = pics;
end
nin = length(x(1, :));
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
    disp(['Doing K=', num2str(nhidden)])
    
    % Use K-fold cross validation
    confusionmat = zeros(2, 2, fold_size);
    for j = 1:fold_size;
        fold_index = fold_indices(j, :);
        % Use j-th block for testing
        x_test = x(fold_index, :);
        t_test = t(fold_index);
        % Use remaining for training
        train_indices = setdiff(1:n_examples, fold_index);
        x_train = x(train_indices, :);
        t_train = t(train_indices)';
        
        net = rbf(nin, nhidden, nout, 'gaussian');
        net = rbfsetbf(net, options, x_train);
        net = rbftrain(net, options, x_train, t_train);
        
        Y = rbffwd(net, x_test)';
        Y(Y >= 0) = 1;
        Y(Y < 0) = -1;
        true_positive = sum(Y == 1 & t_test == 1);
        false_positive = sum(Y == 1 & t_test == -1);
        true_negative = sum(Y == -1 & t_test == -1);
        false_negative = sum(Y == -1 & t_test == 1);
        confusionmat(:, :, j) = [ ...
            true_positive, false_negative;...
            false_positive, true_negative ];
    end
    
    confmat_by_hidden_nodes.matrices(:, :, i) = [ mean(confusionmat(1, 1, :)), mean(confusionmat(1, 2, :)); mean(confusionmat(2, 1, :)), mean(confusionmat(2, 2, :)) ]
end

disp(confmat_by_hidden_nodes)

for i = 1:length(confmat_by_hidden_nodes.nodes);
    nhidden = confmat_by_hidden_nodes.nodes(i);
    accuracy = trace(confmat_by_hidden_nodes.matrices(:, :, i)) / (n_examples / fold_size);
    disp(['Accuracy for ', num2str(nhidden), ' hidden nodes: ', num2str(accuracy)])
end
