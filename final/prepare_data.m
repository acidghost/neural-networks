function [ source, Xtra, Ypaths, Xtes, Xpath, Ypath, end_time ] = prepare_data( data, source, agent_test, use_angles )

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
    
    XYmean = mean([ Xpaths; Xpath; Ypaths; Ypath; source ]);
    Xpaths = Xpaths - repmat(XYmean, size(Xpaths, 1), 1);
    Xpath = Xpath - repmat(XYmean, size(Xpath, 1), 1);
    Ypaths = Ypaths - repmat(XYmean, size(Ypaths, 1), 1);
    Ypath = Ypath - repmat(XYmean, size(Ypath, 1), 1);
    
    source = source - XYmean;

    Xtra_distances = pdist2(Xpaths, source);
    Xtes_distances = pdist2(Xpath, source);
    % mean_distances = mean([Xtra_distances; Xtes_distances]);
    % Xtra_distances = Xtra_distances - mean_distances;
    % Xtes_distances = Xtes_distances - mean_distances;
    if use_angles
        Xtra_angles = compute_angles(Xpaths, source);
        Xtes_angles = compute_angles(Xpath, source);
        % mean_angles = mean([Xtra_angles; Xtes_angles]);
        % Xtra_angles = Xtra_angles - mean_angles;
        % Xtes_angles = Xtes_angles - mean_angles;
        Xtra = [Xpaths Xtra_distances Xtra_angles];
        Xtes = [Xpath Xtes_distances Xtes_angles];
    else
        Xtra = [Xpaths Xtra_distances];
        Xtes = [Xpath Xtes_distances];
    end

end
