clear; close all; clc;

load('pics.mat');

fold_size = 10;
[ n_examples, nin ] = size(pics);
nout = 1;

x = pics;
t = classGlass;

rndperm = randperm(n_examples);
x = x(rndperm, :);
t = t(rndperm);

fold_indices = ass3a_crossvalid(fold_size, n_examples);

confusionmatest = zeros(2, 2, fold_size);
confusionmatrain = zeros(2, 2, fold_size);
for i = 1:fold_size;
    test_indices = fold_indices(i, :);
    train_indices = setdiff(1:n_examples, test_indices);
    x_train = x(train_indices, :);
    t_train = t(train_indices);
    x_test = x(test_indices, :);
    t_test = t(test_indices);
    sig2 = nin;
    gam = n_examples;
    svmmodel = initlssvm(x_train, t_train', 'classification', gam, sig2, 'RBF_kernel');
    svmmodel = trainlssvm(svmmodel);
    
    Yhat = simlssvm(svmmodel, x_test)';
    true_positive = sum(Yhat == 1 & t_test == 1);
    false_positive = sum(Yhat == 1 & t_test == 0);
    true_negative = sum(Yhat == 0 & t_test == 0);
    false_negative = sum(Yhat == 0 & t_test == 1);
    confusionmatest(:, :, i) = [ ...
        true_positive, false_negative;...
        false_positive, true_negative ];
    
    Yhat = simlssvm(svmmodel, x_train)';
    true_positive = sum(Yhat == 1 & t_train == 1);
    false_positive = sum(Yhat == 1 & t_train == 0);
    true_negative = sum(Yhat == 0 & t_train == 0);
    false_negative = sum(Yhat == 0 & t_train == 1);
    confusionmatrain(:, :, i) = [ ...
        true_positive, false_negative;...
        false_positive, true_negative ];
end

avgconfmattest = [mean(confusionmatest(1,1,:)), mean(confusionmatest(1,2,:)); mean(confusionmatest(2,1,:)), mean(confusionmatest(2,2,:))]
accuracytest = trace(avgconfmattest) / (n_examples / fold_size);
disp(['Mean accuracy test: ', num2str(accuracytest)])

avgconfmattrain = [mean(confusionmatrain(1,1,:)), mean(confusionmatrain(1,2,:)); mean(confusionmatrain(2,1,:)), mean(confusionmatrain(2,2,:))]
accuracytrain = trace(avgconfmattrain) / (n_examples / fold_size);
disp(['Mean accuracy train: ', num2str(accuracytrain)])

% svmmodel = initlssvm(x, t', 'classification', [], [], 'RBF_kernel');
% 
% svmmodel = tunelssvm(svmmodel, 'simplex', 'crossvalidatelssvm', { fold_size, 'misclass' });
% 
% svmmodel = trainlssvm(svmmodel);
% 
% Yhat = simlssvm(svmmodel, x);
% 
% accuracy = sum(Yhat == t') / n_examples;
% disp(['Accuracy: ', num2str(accuracy * 100), '%'])
