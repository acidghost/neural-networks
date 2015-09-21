clear; close all; clc;

load('pics.mat');

fold_size = 10;
[ n_examples, nin ] = size(pics);
nout = 1;

x = pics;
t = classGlass;
t(t == 0) = -1;

svmmodel = initlssvm(x, t', 'classification', [], [], 'RBF_kernel');

svmmodel = tunelssvm(svmmodel, 'simplex', 'crossvalidatelssvm', { fold_size, 'misclass' });

svmmodel = trainlssvm(svmmodel);

Yhat = simlssvm(svmmodel, x);

accuracy = sum(Yhat == t') / n_examples;
disp(['Accuracy: ', num2str(accuracy * 100), '%'])
