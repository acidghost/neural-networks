function [ data ] = generate_agents( data, source, theta )

    end_time = size(data,1);
    number_of_agents = size(data,2)/2;
    
    new_agents = zeros(end_time, number_of_agents);
    for agent = 1:number_of_agents;
        agent_path = [data(:, (2*agent)-1), data(:, 2*agent)];
        new_agents(:, [(2*agent)-1, 2*agent]) = rotateVectorPoint(agent_path, theta, source);
    end
    
    data = [data, new_agents];
    
end

