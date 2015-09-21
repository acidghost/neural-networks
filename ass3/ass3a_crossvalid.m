function [ fold_indices ] = ass3a_crossvalid( fold_size, n_examples )

    % Calculate indices for K-fold cross validation
    fold_indices = zeros(fold_size, floor(n_examples / fold_size));
    for i = 1:fold_size;
        if i == 1
            fold_indices(i, :) = 1:floor(n_examples / fold_size);
        else
            last_index = fold_indices(i - 1, floor(n_examples / fold_size));
            fold_indices(i, :) = (last_index + 1):(last_index + floor(n_examples / fold_size));
        end
    end

end

