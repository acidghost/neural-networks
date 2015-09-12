clear; clf

sep_points = 1000;
scale = 20;
separable = get_data(sep_points, scale);

not_separable = importdata('two_class_example_not_separable.dat');
not_sep_points = size(not_separable, 1);

learning_rate = 0.001;
max_iterations = 1000;

% Apply perceptron to both
sep_weights = perceptron(separable, learning_rate, max_iterations);
not_sep_weights = perceptron(not_separable, learning_rate, max_iterations);

% Test weights and measure error for perceptron
sep_right = testweights(separable, sep_weights');
not_sep_right = testweights(not_separable, not_sep_weights');
fprintf('%d / %d separable perceptron\n', sep_right, sep_points)
fprintf('%d / %d not separable perceptron\n', not_sep_right, not_sep_points)

% Plot data and decision boundaries
subplot(1, 2, 1)
scatterboundary(separable, sep_weights)
subplot(1, 2, 2)
scatterboundary(not_separable, not_sep_weights);


% Apply LMS to both
sep_glm_options = [ zeros(13, 1); max_iterations ];
separable_glm = glm(2, 1, 'linear', learning_rate);
separable_glm = glmtrain(separable_glm, sep_glm_options, separable(:, 1:2), separable(:, 3));
separable_glm_weights = [ separable_glm.b1 separable_glm.w1' ];

not_sep_glm_options = [ zeros(13, 1); max_iterations ];
not_separable_glm = glm(2, 1, 'linear', learning_rate);
not_separable_glm = glmtrain(not_separable_glm, not_sep_glm_options, not_separable(:, 1:2), not_separable(:, 3));
not_sep_glm_weights = [ not_separable_glm.b1 not_separable_glm.w1' ];

% Test weights and measure error for LMS
sep_right_glm = testweights(separable, separable_glm_weights);
not_sep_right_glm = testweights(not_separable, not_sep_glm_weights);
fprintf('%d / %d separable GLM\n', sep_right_glm, sep_points)
fprintf('%d / %d not separable GLM\n', not_sep_right_glm, not_sep_points)

% Plot LMS decision boundaries
figure
subplot(1, 2, 1)
scatterboundary(separable, separable_glm_weights)
subplot(1, 2, 2)
scatterboundary(not_separable, not_sep_glm_weights);
