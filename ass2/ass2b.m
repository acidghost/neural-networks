clear; clf

sep_points = 1000;
scale = 20;
separable = get_data(sep_points, scale);

not_separable = importdata('two_class_example_not_separable.dat');
not_sep_points = size(not_separable, 1);

learning_rate = 0.1;
max_iterations = 10000;

% Apply perceptron to both
[sep_weights, sep_errors, sep_epoch] = perceptron(separable, learning_rate, max_iterations);
[not_sep_weights, not_sep_errors, not_sep_epoch] = perceptron(not_separable, learning_rate, max_iterations);

% Test weights and measure error for perceptron
sep_right = testweights(separable, sep_weights');
not_sep_right = testweights(not_separable, not_sep_weights');
fprintf('%d / %d separable perceptron\n', sep_right, sep_points)
fprintf('%d / %d not separable perceptron\n', not_sep_right, not_sep_points)

% Plot data and decision boundaries
axis_sep = [-1 scale+1 -1 scale+1];
axis_not_sep = [...
    min(not_separable(:, 1))-.1 max(not_separable(:, 1))+.1...
    min(not_separable(:, 2))-.1 max(not_separable(:, 2))+.1...
];
subplot(2, 2, 1)
scatterboundary(separable, sep_weights)
axis(axis_sep)
title('Separable perceptron')
subplot(2, 2, 2)
scatterboundary(not_separable, not_sep_weights)
axis(axis_not_sep)
title('Not separable perceptron')


% Apply LMS to both
glm_options = [ zeros(13, 1); max_iterations ];
separable_glm = glm(2, 1, 'linear', learning_rate);
separable_glm = glmtrain(separable_glm, glm_options, separable(:, 1:2), separable(:, 3));
separable_glm_weights = [ separable_glm.b1 separable_glm.w1' ];

not_separable_glm = glm(2, 1, 'linear', learning_rate);
not_separable_glm = glmtrain(not_separable_glm, glm_options, not_separable(:, 1:2), not_separable(:, 3));
not_sep_glm_weights = [ not_separable_glm.b1 not_separable_glm.w1' ];

% Test weights and measure error for LMS
sep_right_glm = testweights(separable, separable_glm_weights);
not_sep_right_glm = testweights(not_separable, not_sep_glm_weights);
fprintf('%d / %d separable GLM\n', sep_right_glm, sep_points)
fprintf('%d / %d not separable GLM\n', not_sep_right_glm, not_sep_points)

% Plot LMS decision boundaries
subplot(2, 2, 3)
scatterboundary(separable, separable_glm_weights)
axis(axis_sep)
title('Separable GLM')
subplot(2, 2, 4)
scatterboundary(not_separable, not_sep_glm_weights)
axis(axis_not_sep)
title('Not separable GLM')
