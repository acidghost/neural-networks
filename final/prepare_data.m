function [ Xtra, Ypaths, Xtes, Xpath, Ypath, end_time ] = prepare_data( data, source, agent_test )

    end_time = size(data,1);
    number_of_agents = size(data,2)/2;
    
    data = data - mean(mean(data));
    % data = detrend(data);
    
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

end

