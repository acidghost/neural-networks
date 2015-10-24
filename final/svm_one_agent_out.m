function [ model, Xtes, Xpath, Ypath, RMSE ] = svm_one_agent_out( data, source, agent_test )

    fprintf('Doing test agent %d\n', agent_test);
    
    [ Xtra, Ypaths, Xtes, Xpath, Ypath, end_time ] = prepare_data( data, source, agent_test );

    model = initlssvm( Xtra, Ypaths, 'f', [], [], 'RBF_kernel' );
    model = tunelssvm(model, 'simplex', 'crossvalidatelssvm', { length(agents_tra), 'mae' });
    model = trainlssvm(model);
    
    % Yhs = simlssvm(model, Xtes);
    
    % Predict path starting from first position
    fullpath = [Xpath; Ypath(end_time-1, :)];
    Xs = Xtes(1, :);
    Yhs = [Xs(1, 1:2); zeros(end_time-1, 2)];
    for i = 2:end_time;
        Yh = simlssvm(model, Xs);
        Yhs(i, :) = Yh;
        Xs = [Yh pdist2(Yh, source)];
    end
    error = diag(pdist2(fullpath, Yhs));
    RMSE = sqrt(mean(error.^2));

    fprintf('\n\nRMSE for agent test %d: %f\n', agent_test, RMSE);

end

