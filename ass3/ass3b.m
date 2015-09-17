clear all; close all; clc;
load('pics.mat');

% preprocess data
x = pics - ones(400, 2576) * mean(mean(pics));
t = classGlass;
yg = zeros(1, 400);
nhidden = [1, 2, 3, 4, 5, 6, 15, 25, 50, 100, 2576];
percentages_correct = zeros(10, 1);
confusion_matrices = zeros(2, 2, length(nhidden));
for j = 1:length(nhidden)
    %10 fold cross-validation
    for i = 1:10
        test_values = 40 * (i - 1) + 1:40 * i;
        train_values = (1: 1: 400);
        train_values(test_values) = [];
        x_training = x(train_values, :);
        x_test = x(test_values, :);
        t_train = t(1, train_values)';
        t_test = t(1, test_values)';

        nout = 1;
        ncycles = 20; 
        net = mlp(2576, nhidden(j), nout, 'logistic');
        options = foptions;
        options(1) = 1;
        options(14) = ncycles;
        options(18) = 0.01;

        [net] = netopt(net, options, x_training, t_train, 'graddesc'); % quasinewton computationally very expensive when using large numbers of hidden nodes
        yg(test_values) = mlpfwd(net, x_test);
    end
    
    msp = mean((t - yg) .^ 2);
    y_round = round(yg);
    correct_positive = sum((y_round == 1) & (t == 1));
    false_positive = sum((y_round == 1) & (t == 0));
    correct_negative = sum((y_round == 0) & (t == 0));
    false_negative = sum((y_round == 0) & (t == 1));
    confusion_matrix = [correct_positive, false_positive; false_negative, correct_negative];
    confusion_matrices(:, :, j) = confusion_matrix;
    percentages_correct(j) = (correct_positive + correct_negative) / (correct_positive + false_positive + correct_negative + false_negative);
end

scatter(nhidden, percentages_correct);
disp(confusion_matrices);
max(percentages_correct)
