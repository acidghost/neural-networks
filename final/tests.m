clear; clf;

load 'dataset_final_assignment.mat';

% grid size
max_x = 600;
max_y = 800;

end_time = size(data,1);

number_of_agents = size(data,2)/2;

% pre-process data (mirror all people over y-axis, !only if not already done!)
if data(1,2) > max_y/2
  for i = 1:end_time
    for a = 1:size(data(1,:),2)/2
      data(i,2*a) = max_y-data(i,2*a);
    end
  end
end

% sources of panic (e.g. shouting individual)
source = [542.0, max_y-439.0];

% % draw person 'a'
% a = 8;
% hold on
% grid on
% axis([0 max_x+1 0 max_y+1])
% path = zeros(end_time, 2);
% for i = 1:end_time;
%     plot(data(i,(2*a)-1),data(i,(2*a)),'Color',[0 0 1],'Marker','.','MarkerSize',10);
%     path(i, :) = [data(i, (2*a)-1), data(i, 2*a)];
% end
% 
% figure
% plot(path(:, 1), path(:, 2))

RMSEs = zeros(number_of_agents, 1);
for agent_test = 1:number_of_agents;
    [model, Xtes, Xpath, Ypath, RMSE] = svm_one_agent_out(data, source, agent_test);
    RMSEs(agent_test) = RMSE;
end
