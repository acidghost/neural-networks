function [ net, source, fullpath, Yhs, RMSE ] = rbf_one_agent_out( data, source, agent_test, use_angles, nhidden, kmeans_init )

    % fprintf('Doing test agent %d\n', agent_test);
    
    [ source, Xtra, Ypaths, Xtes, Xpath, Ypath, end_time ] = prepare_data( data, source, agent_test, use_angles );
    
    switch nargin
        case 4
            nhidden = size(Xtra, 1) / (end_time-1); %length(Xtra);
            nhidden = nhidden * 2;
            kmeans_init = 'random';
        case 5
            kmeans_init = 'random';
    end
    
    nin = size(Xtra, 2);
    nout = size(Xpath, 2);
    options = zeros(1, 14);
    options(1, 1) = 1;      % uneffective
    options(1, 5) = 1;
    options(1, 14) = 100;   % uneffective
    net = rbf(nin, nhidden, nout, 'gaussian');
    % net.alpha = 0.3;
    net = rbfsetbf(net, options, Xtra, kmeans_init);
    net = rbftrain(net, options, Xtra, Ypaths);
    
    RMSE = zeros(1, 3);
    
    % Predict path starting from first position
    fullpath = [Xpath; Ypath(end_time-1, :)];
    Xs = Xtes(1, :);
    Yhs = [Xs(1, 1:2); zeros(end_time-1, 2)];
    for i = 2:end_time;
        Yh = rbffwd(net, Xs);
        Yhs(i, :) = Yh;
        if use_angles
            Xs = [Yh pdist2(Yh, source) compute_angles(Yh, source)];
        else
            Xs = [Yh pdist2(Yh, source)];
        end
    end
    error = diag(pdist2(fullpath, Yhs));
    RMSE(1, 1) = sqrt(mean(error.^2));
    
    % Predict only next step
    errors = zeros(end_time-1, 1);
    for i = 1:end_time-1;
        Xs = Xtes(i, :);
        Yh = rbffwd(net, Xs);
        errors(i) = pdist2(fullpath(i+1, :), Yh);
    end
    RMSE(1, 2) = sqrt(mean(errors.^2));
    
    % Predict next three steps
    errors = zeros(end_time-3, 1);
    for i = 1:end_time-3;
        Xs = Xtes(i, :);
        local_errors = zeros(3, 1);
        for j = 1:3;
            Yh = rbffwd(net, Xs);
            local_errors(j) = pdist2(fullpath(i+j, :), Yh);
            if use_angles
                Xs = [ Yh, pdist2(source, Yh), compute_angles(Yh, source) ];
            else
                Xs = [ Yh, pdist2(source, Yh) ];
            end
        end
        errors(i) = sum(local_errors);
    end
    RMSE(1, 3) = sqrt(mean(errors.^2));

    % fprintf('RMSE for agent test %d: %f %f %f\n', agent_test, RMSE);

end

