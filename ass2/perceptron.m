function [weights, errors, iteration] = perceptron( dataset, learning_rate, iterations )

    n = size(dataset);
    data_size = n(1);
    features = n(2)-1;
    class_index = features + 1;
    classes = unique(dataset(:, class_index));
    weights = zeros(features + 1, 1);
    dataset = [ ones(data_size, 1) dataset ];
    class_index = class_index + 1;
    
    errors = zeros(iterations, 1);
    for iteration = 1:iterations;
      error = 0;
      for i = 1:data_size
        input = transpose(dataset(i, 1:features + 1));
        y = sign(transpose(weights) * input);
        if dataset(i, class_index) == classes(1);
            d = -1;
        else
            d = 1;
        end
        
        error = error + (d - y)^2;
        
        weights = weights + learning_rate * (d - y) * input;
      end
      errors(iteration, 1) = error / data_size;
      if errors(iteration, 1) == 0
        return;
      end
    end

end

