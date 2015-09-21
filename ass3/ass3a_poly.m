function [ error_by_degree ] = ass3a_poly( data, fold_size )

    % Shuffle data for cross validation
    permutation = randperm(length(data.x));
    x = data.x(permutation);
    t = data.t(permutation);
    
    % Preprocessing
    x = x - mean(x);
    t = t - mean(t);
    
    figure
    axis_values = [ min(x) max(x) min(t) max(t) ];

    % polyfit_degrees = [1, 2, 3, 4, 5, 10, 15, 20];
    polyfit_degrees = [2, 3, 4, 5, 7, 10, 15, 25];
    
    % Calculate indices for K-fold cross validation
    fold_indices = ass3a_crossvalid(fold_size, length(x));
    
    % Try different degrees for polyfit
    error_by_degree = [ polyfit_degrees' zeros(length(polyfit_degrees), 2) ];
    for i = 1:length(polyfit_degrees);
        degree = polyfit_degrees(i);
        subplot(length(polyfit_degrees) / 2, 2, i)
        hold on
        scatter(x, t)
        % Use K-fold cross validation
        % first column is for error on training set
        % second is for error on test set
        RMSE = zeros(fold_size, 2);
        for j = 1:fold_size;
            fold_index = fold_indices(j, :);
            % Use j-th block for testing
            x_test = x(fold_index);
            t_test = t(fold_index);
            % Use remaining for training
            train_indices = setdiff(1:length(x), fold_index);
            x_train = x(train_indices);
            t_train = t(train_indices);

            % Apply polyfit
            p = polyfit(x_train, t_train, degree);

            % Get error against training
            Y = polyval(p, x_train);
            RMSE(j, 1) = sqrt(mean((Y - t_train).^2));

            % Get error against test set
            Y = polyval(p, x_test);
            RMSE(j, 2) = sqrt(mean((Y - t_test).^2));

            % Plot regression line
            plot(x, polyval(p, x))
        end
        
        legendstr = cellstr(num2str((1:fold_size)'))';
        legend([ 'data' legendstr ], 'Location', 'southeast')
        title(['Polyfit with degree ', num2str(degree)])
        axis(axis_values)
        hold off
        error_by_degree(i, 2:3) = mean(RMSE);
    end
    
end
