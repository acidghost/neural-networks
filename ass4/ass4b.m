clear; close all; clc;

n_examples = 1000;
scale = 10;
dataset = get_data(n_examples, scale);

lssvmparams = { dataset(:, 1:2), dataset(:, 3), 'classification', 10, .4 };

[alpha, b] = trainlssvm(lssvmparams);

lssvmout = simlssvm(lssvmparams, { alpha, b }, dataset(:, 1:2));

correct = sum(lssvmout == dataset(:, 3));
disp(['Correctly classified: ', num2str(correct)])

plotlssvm(lssvmparams, { alpha, b })
