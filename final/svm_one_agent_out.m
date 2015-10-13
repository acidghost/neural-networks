function [ model, Xtes, Xpath, Ypath, RMSE ] = svm_one_agent_out( data, source, agent_test )

    fprintf('Doing test agent %d\n', agent_test);
    
    end_time = size(data,1);
    number_of_agents = size(data,2)/2;
    
    agents_tra = setdiff(1:number_of_agents, agent_test);
    Xpaths = zeros((end_time-1) * length(agents_tra), 2);
    Ypaths = zeros((end_time-1) * length(agents_tra), 2);
    Xcounter = 1;
    Ycounter = 1;
    for agent = agents_tra;
        for i = 1:end_time;
            if i == 1
                Xpaths(Xcounter, :) = [data(i, (2*agent)-1), data(i, 2*agent)];
                Xcounter = Xcounter + 1;
            elseif i == end_time;
                Ypaths(Ycounter, :) = [data(i, (2*agent)-1), data(i, 2*agent)];
                Ycounter = Ycounter + 1;
            else
                Xpaths(Xcounter, :) = [data(i, (2*agent)-1), data(i, 2*agent)];
                Xcounter = Xcounter + 1;
                Ypaths(Ycounter, :) = [data(i, (2*agent)-1), data(i, 2*agent)];
                Ycounter = Ycounter + 1;
            end
        end
    end
    Xpath = zeros(end_time - 1, 2);
    Ypath = zeros(end_time - 1, 2);
    for i = 1:end_time-1;
        Xpath(i, :) = [ data(i, (2*agent_test)-1), data(i, 2*agent_test) ];
        Ypath(i, :) = [ data(i+1, (2*agent_test)-1), data(i+1, 2*agent_test) ];
    end

    Xtra = [Xpaths pdist2(Xpaths, source)];
    Xtes = [Xpath pdist2(Xpath, source)];

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

