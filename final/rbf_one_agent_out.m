function [ net, Xtes, Xpath, Ypath, RMSE ] = rbf_one_agent_out( data, source, agent_test, nhidden, kmeans_init )

    % fprintf('Doing test agent %d\n', agent_test);
    
    [ Xtra, Ypaths, Xtes, Xpath, Ypath, end_time ] = prepare_data( data, source, agent_test );
    
    switch nargin
        case 3
            nhidden = size(Xtra, 1) / (end_time-1); %length(Xtra);
            nhidden = nhidden * 2;
            kmeans_init = 'random';
        case 4
            kmeans_init = 'random';
    end
    
    nin = size(Xtra, 2);
    nout = size(Xpath, 2);
    options = zeros(14);
    options(5) = 1;
    net = rbf(nin, nhidden, nout, 'gaussian');
    net = rbfsetbf(net, options, Xtra, kmeans_init);
    net = rbftrain(net, options, Xtra, Ypaths);
    
    % Predict path starting from first position
    fullpath = [Xpath; Ypath(end_time-1, :)];
    Xs = Xtes(1, :);
    Yhs = [Xs(1, 1:2); zeros(end_time-1, 2)];
    for i = 2:end_time;
        Yh = rbffwd(net, Xs);
        Yhs(i, :) = Yh;
        Xs = [Yh pdist2(Yh, source)];
    end
    error = diag(pdist2(fullpath, Yhs));
    RMSE = sqrt(mean(error.^2));

    % fprintf('\n\nRMSE for agent test %d: %f\n', agent_test, RMSE);

end

