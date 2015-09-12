function [ priors, conditionals, posteriors ] = calc_probabilities( dataset )

    classes = size(dataset, 1);
    values = size(dataset, 2);
    total_examples = sum(sum(dataset));
    
    priors = zeros(classes, 1);
    conditionals = zeros(classes, values);
    posteriors = zeros(classes, values);
    scaling_factors = zeros(1, values);
    for i = 1:classes;
        row_sum = sum(dataset(i, :));
        priors(i, 1) = row_sum / total_examples;
        for j = 1:values;
            conditionals(i, j) = dataset(i, j) / row_sum;
        end
    end
    
    for i = 1:classes;
        for j = 1:values;
            scaling_factors(1, j) = conditionals(1, j) * priors(1, 1) + conditionals(2, j) * priors(2, 1);
            posteriors(i, j) = (priors(i, 1) * conditionals(i, j)) / scaling_factors(1, j);
        end
    end

end

