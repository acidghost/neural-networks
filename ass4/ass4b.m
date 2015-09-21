clear; close all; clc;

n_examples = 1000;
scale = 10;
dataset = get_data(n_examples, scale);

p_or_not = {'preprocess' 'original'};

for i = 1:length(p_or_not);
    p = p_or_not{i};

    lssvmparams = { dataset(:, 1:2), dataset(:, 3), 'classification', 1, 0.5, 'RBF_kernel', p };

    [alpha, b] = trainlssvm(lssvmparams);

    lssvmout = simlssvm(lssvmparams, { alpha, b }, dataset(:, 1:2));

    correct = sum(lssvmout == dataset(:, 3));
    disp([p, ' correctly classified: ', num2str(correct)])

    % subplot(1, 2, i)
    figure
    plotlssvm(lssvmparams, { alpha, b })
end
