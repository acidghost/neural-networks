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

% Do SVM
% RMSEs = zeros(number_of_agents, 1);
% for agent_test = 1:number_of_agents;
%     [model, Xtes, Xpath, Ypath, RMSE] = svm_one_agent_out(data, source, agent_test);
%     RMSEs(agent_test) = RMSE;
% end

% Do RBF
hnodes = 4:4:100;
meanRMSE = [hnodes', zeros(length(hnodes), 1)];
rbftrials = 10;
for nhidden = hnodes;
    RMSEs = zeros(number_of_agents, 1);
    for agent_test = 1:number_of_agents;
        % Do some trials to cope with kmeans noise
        trials = zeros(rbftrials, 1);
        for t = 1:rbftrials;
            [net, Xtes, Xpath, Ypath, RMSE] = rbf_one_agent_out(data, source, agent_test, nhidden);
            trials(t) = RMSE;
        end
        RMSEs(agent_test) = mean(trials);
    end
    m = mean(RMSEs);
    fprintf('Mean RMSE for %d hidden units: %f\n', nhidden, m);
    meanRMSE(meanRMSE(:,1) == nhidden, 2) = m;
end

[sortedmeanRMSE, indexes] = sort(meanRMSE(:, 2));
fprintf('Ranked hidden layer size (mean over %d trials)\n', rbftrials);
disp(table(meanRMSE(indexes, 1), meanRMSE(indexes, 2), 'VariableNames', {'LayerSize', 'MeanRMSE'}))
