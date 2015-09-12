clear; clf;

%% Generate a linearly separable dataset of points
points = 1000;
scale = 20;
dataset = get_data(points, scale);

%% Execute Perceptron Convergence Theorem
[weights, errors, epoch] = perceptron(dataset, 0.1, 100);

%% Test the learned weights
right = testweights(dataset, weights');
if right == points
    fprintf('All points correctly classified\n');
else
    fprintf('%d points were classified correctly\n', right);
end

% Plot dataset and decision boundary
scatterboundary(dataset, weights)

%% Test different learning rates...
trials = 1000;
learning_rates = zeros(trials, 1);
epochs = zeros(trials, 1);
errors_sums = zeros(trials, 1);
index = 1;
for rate = linspace(0.01, 1, trials);
  [w, e, epochs(index, 1)] = perceptron(dataset, rate, 100);
  %fprintf('Learning rate %f epoch %d\n', rate, epoch);
  learning_rates(index, 1) = rate;
  errors_sums(index, 1) = sum(e);
  index = index + 1;
end

% ... and then plot its relation with 
% the overall error and number of epochs
figure; hold on;
plot(learning_rates, epochs, learning_rates, errors_sums);
xlabel('Learning rate')
legend('Epochs', 'Error')
hold off;

%% Import not separable dataset
not_sep = importdata('two_class_example_not_separable.dat');

% Run perceptron
[weights, errors, epoch] = perceptron( not_sep, 0.9, 1000 );
fprintf('Non separable epochs: %d\n', epoch);

figure
scatterboundary(not_sep, weights)

%% Test the learned weights for not separable case
right = testweights(not_sep, weights');
if right == points
    fprintf('not_sep: All %d points correctly classified\n', size(not_sep, 1));
else
    fprintf('not_sep: %d / %d points were classified correctly\n', right, size(not_sep, 1));
end

%% Test different learning rates with not separable data
trials = 1000;
learning_rates = zeros(trials, 1);
epochs = zeros(trials, 1);
errors_sums = zeros(trials, 1);
index = 1;
for rate = linspace(0.01, 1, trials);
    [w, e, epochs(index, 1)] = perceptron(not_sep, rate, 100);
    %fprintf('Learning rate %f epoch %d\n', rate, epoch);
    learning_rates(index, 1) = rate;
    errors_sums(index, 1) = sum(e);
    index = index + 1;
end
figure; hold on;
plot(learning_rates, epochs, learning_rates, errors_sums);
xlabel('Learning rate')
legend('Epochs', 'Error')
hold off;
