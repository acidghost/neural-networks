clear; clf;

size_dataset = 1000;
dataset = get_data(size_dataset, 20);
glm_datset = glm(2, 1, 'linear');
glm_datset = glmtrain(glm_datset, zeros(14), dataset(:, 1:2), dataset(:, 3));
w_dataset = [ glm_datset.b1 glm_datset.w1' ];

subplot(1, 2, 1)
scatterboundary(dataset, w_dataset)

e_dataset = testweights(dataset, w_dataset);
fprintf('Separable case: %d / %d correct\n', e_dataset, size_dataset)




not_sep = importdata('two_class_example_not_separable.dat');
size_not_sep = size(not_sep, 1);
glm_not_sep = glm(2, 1, 'linear');
glm_not_sep = glmtrain(glm_not_sep, zeros(14), not_sep(:, 1:2), not_sep(:, 3));
w_not_sep = [ glm_not_sep.b1 glm_not_sep.w1' ];

subplot(1, 2, 2)
scatterboundary(not_sep, w_not_sep)

e_not_sep = testweights(not_sep, w_not_sep);
fprintf('Not separable case: %d / %d correct\n', e_not_sep, size_not_sep)


% Test different parameter settings for LMS...
