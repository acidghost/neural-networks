clear; clf;

samples = 1000;
values = 16;
x = 1:values;
dataset = gen_dataset(samples, x);
subplot(2, 2, 1);
bar(x, dataset');
legend('b', 'a');
title(sprintf('Generated dataset with %d samples', samples));
subplot(2, 2, 2);
plot(x, dataset(1,:), x, dataset(2,:));
legend('b', 'a');

[ priors, conditionals, posteriors ] = calc_probabilities(dataset);

subplot(2, 2, 3);
bar(x, posteriors');
legend('b', 'a');
title('Posterior probabilities histogram');
subplot(2, 2, 4);
plot(x, posteriors(1,:), x, posteriors(2,:));
legend('b', 'a');

boundaries_with_errors = calc_boundaries_errors(posteriors);
boundaries_with_total_errors = sum(boundaries_with_errors);

figure;
x = 1:values-1;
subplot(1, 2, 1);
plot(x, boundaries_with_errors(1,:), x, boundaries_with_errors(2,:));
legend('b', 'a');
title('Error per class by decision boundary');
xlabel('Decision boundary');
ylabel('Error');
subplot(1, 2, 2);
plot(x, boundaries_with_total_errors);
title('Total error by decision boundary');
xlabel('Decision boundary');
ylabel('Total error');

% Find the decision boundary that minimizes the total error
[ minimal_total_error, minimal_total_error_boundary ] = min(boundaries_with_total_errors);
minimal_total_error_boundary = minimal_total_error_boundary - 1;
fprintf('Found an optimal decision boundary between %d and %d with error %f\n',...
    minimal_total_error_boundary, minimal_total_error_boundary + 1, minimal_total_error);

% Now increase the cost for misclassifying a 'b' as an 'a' than the opposite
new_boundaries_with_errors = boundaries_with_errors;
new_boundaries_with_errors(2, :) = 3 * new_boundaries_with_errors(2, :);
new_boundaries_with_total_errors = sum(new_boundaries_with_errors);
% and find the new decision boundary that minimizes the total error
[ new_minimal_total_error, new_minimal_total_error_boundary ] = min(new_boundaries_with_total_errors);
new_minimal_total_error_boundary = new_minimal_total_error_boundary - 1;
fprintf('Found an optimal decision boundary between %d and %d with error %f\n',...
    new_minimal_total_error_boundary, new_minimal_total_error_boundary + 1, new_minimal_total_error);

figure;
subplot(1, 2, 1);
x = 1:values-1;
plot(x, new_boundaries_with_errors(1,:), x, new_boundaries_with_errors(2,:));
legend('b', 'a');
title('New error per class by decision boundary');
xlabel('Decision boundary');
ylabel('Error');
subplot(1, 2, 2);
plot(x, new_boundaries_with_total_errors);
title('New total error by decision boundary');
xlabel('Decision boundary');
ylabel('Total error');

drawnow;
