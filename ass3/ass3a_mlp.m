function [ error_by_hidden_nodes ] = ass3a_mlp( data, fold_size, backprop, func, alpha, cycles, conv_threshold )
    switch nargin
        case 2
            backprop = true;
            func = 'linear';
            alpha = 0.01;
            cycles = 300;
            conv_threshold = 0.001;
        case 3
            func = 'linear';
            alpha = 0.01;
            cycles = 300;
            conv_threshold = 0.001;
        case 4
            alpha = 0.01;
            cycles = 300;
            conv_threshold = 0.001;
        case 5
            cycles = 300;
            conv_threshold = 0.001;
        case 6
            conv_threshold = 0.001;
        otherwise
            error('Wrong number of arguments')
    end
    
    n_examples = length(data.x);

    % Shuffle data for cross validation
    permutation = randperm(n_examples);
    x = data.x(permutation);
    t = data.t(permutation);
    
    % Preprocessing
    x = x - mean(x);
    t = t - mean(t);
    
    figure
    axis_values = [ min(x) max(x) min(t) max(t) ];

    number_of_nodes = [2, 3, 4, 5, 7, 10, 15, 25];

    % Calculate indices for K-fold cross validation
    fold_indices = ass3a_crossvalid(fold_size, n_examples);
    
    % Set parameters
    nin = 1;
    nout = 1;
    
    % Try different number of hidden nodes of MLP
    error_by_hidden_nodes = [ number_of_nodes' zeros(length(number_of_nodes), 2) ];
    for i = 1:length(number_of_nodes);
        nhidden = number_of_nodes(i);
        subplot(length(number_of_nodes) / 2, 2, i)
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
            train_indices = setdiff(1:n_examples, fold_index);
            x_train = x(train_indices);
            t_train = t(train_indices);

            % Init network
            net = mlp(nin, nhidden, nout, func);
            if backprop
                for k = 1:cycles;
                    % Feed-forward the inputs through the network
                    [Y, Z] = mlpfwd(net, x_train);

                    % Back-propagate the error
                    G = mlpbkp(net, x_train, Z, Y - t_train);

                    % Update weights in network
                    old_weights = netpak(net);
                    weights = old_weights - alpha * G;
                    net = netunpak(net, weights);

                    % Check if convergence is reached
                    if length(G(G < conv_threshold)) == length(G)
                        break;
                    end
                end
            else
                % net = netopt(net, [1; zeros(17, 1)], x_train, t_train, 'graddesc');
                net = mlptrain(net, x_train, t_train, cycles);
            end

            Yhat = zeros(n_examples, 1);

            % Get error against training
            Y = mlpfwd(net, x_train);
            Yhat(train_indices) = Y;
            RMSE(j, 1) = sqrt(mean((Y - t_train).^2));

            % Get error against test set
            Y = mlpfwd(net, x_test);
            Yhat(fold_index) = Y;
            RMSE(j, 2) = sqrt(mean((Y - t_test).^2));

            % Plot regression line
            [xsorted, I] = sort(x);
            ysorted = Yhat(I);
            plot(xsorted, ysorted)
        end
        
        % Find regression line against all data
%         net = mlp(nin, nhidden, nout, func);
%         for k = 1:cycles;
%             [Y, Z] = mlpfwd(net, x);
% 
%             G = mlpbkp(net, x, Z, Y - t);
% 
%             old_weights = netpak(net);
%             weights = old_weights - alpha * G;
%             net = netunpak(net, weights);
% 
%             if length(G(G < conv_threshold)) == length(G)
%                 break;
%             end
%         end
%         Yhat = mlpfwd(net, x);
%         [xsorted, I] = sort(x);
%         ysorted = Yhat(I);
%         plot(xsorted, ysorted)
        
        legendstr = cellstr(num2str((1:fold_size)'))';
        legend([ 'data' legendstr ], 'Location', 'southeast')
        title(['MLP with ', num2str(number_of_nodes(i)), ' hidden nodes'])
        axis(axis_values)
        hold off
        error_by_hidden_nodes(i, 2:3) = mean(RMSE);
    end
        
end
