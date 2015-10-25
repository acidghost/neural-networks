clear; clf;

load 'dataset_final_assignment.mat';

% grid size
max_x = 600;
max_y = 800;

end_time = size(data,1);

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

% Generate new data
data = generate_agents(data, source, 0.05);

number_of_agents = size(data,2)/2;

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
errors_returned = 3;
hnodes = 4:4:100;
% hnodes = 4:4:12;
rbftrials = 5;
use_angles_opts = [true, false];
meanRMSEs = [hnodes', zeros(length(hnodes), 2 * errors_returned)];
for i = 1:length(use_angles_opts);
    use_angles = use_angles_opts(i);
    for nhidden = hnodes;
        RMSEs = zeros(number_of_agents, errors_returned);
        for agent_test = 1:number_of_agents;
            % Do some trials to cope with kmeans noise
            trials = zeros(rbftrials, errors_returned);
            for t = 1:rbftrials;
                [ net, source, fullpath, Yhs, RMSE ] = rbf_one_agent_out(data, source, agent_test, use_angles, nhidden);
                trials(t, :) = RMSE;
            end
            RMSEs(agent_test, :) = mean(trials);
        end
        m = mean(RMSEs);
        fprintf('Mean RMSE for %d hidden units, using angles %i: %f %f %f\n', nhidden, use_angles, m);
        meanRMSEs(meanRMSEs(:,1) == nhidden, 2+errors_returned*(i-1):1+errors_returned+errors_returned*(i-1)) = m;
    end
end

for i = 1:length(use_angles_opts);
    [sortedmeanRMSE, indexes] = sort(meanRMSEs(:, 2+errors_returned*(i-1)));
    fprintf('Ranked hidden layer size (mean over %d trials, using angles %i)\n', rbftrials, use_angles_opts(i));
    disp(table(meanRMSEs(indexes, 1), meanRMSEs(indexes, 2+errors_returned*(i-1)), 'VariableNames', {'LayerSize', 'MeanRMSE'}))
end

axis_range = [min(hnodes), max(hnodes), 0, max(max(meanRMSEs(:, 2:size(meanRMSEs, 2))))];
subplot(1, 3, 1)
hold on; box on
plot(meanRMSEs(:, 1), meanRMSEs(:, 2), 'r-')
plot(meanRMSEs(:, 1), meanRMSEs(:, 2+errors_returned), 'b-')
legend('W angles', 'W/o angles')
xlabel('Number of hidden nodes')
ylabel('Mean error')
axis(axis_range)
title('Full path prediction')
subplot(1, 3, 2)
hold on; box on
plot(meanRMSEs(:, 1), meanRMSEs(:, 4), 'r-')
plot(meanRMSEs(:, 1), meanRMSEs(:, 4+errors_returned), 'b-')
legend('W angles', 'W/o angles')
xlabel('Number of hidden nodes')
ylabel('Mean error')
axis(axis_range)
title('Predict next three step')
subplot(1, 3, 3)
hold on; box on
plot(meanRMSEs(:, 1), meanRMSEs(:, 3), 'r-')
plot(meanRMSEs(:, 1), meanRMSEs(:, 3+errors_returned), 'b-')
legend('W angles', 'W/o angles')
xlabel('Number of hidden nodes')
ylabel('Mean error')
axis(axis_range)
title('Predict next steps')

figure
axis_range = [min(hnodes(2:length(hnodes))), max(hnodes(2:length(hnodes))),...
    0, max(max(meanRMSEs(2:size(meanRMSEs, 1), 2:size(meanRMSEs, 2))))];
subplot(1, 3, 1)
hold on; box on
plot(meanRMSEs(2:size(meanRMSEs, 1), 1), meanRMSEs(2:size(meanRMSEs, 1), 2), 'r-')
plot(meanRMSEs(2:size(meanRMSEs, 1), 1), meanRMSEs(2:size(meanRMSEs, 1), 2+errors_returned), 'b-')
legend('W angles', 'W/o angles')
xlabel('Number of hidden nodes')
ylabel('Mean error')
axis(axis_range)
title('Full path prediction')
subplot(1, 3, 2)
hold on; box on
plot(meanRMSEs(2:size(meanRMSEs, 1), 1), meanRMSEs(2:size(meanRMSEs, 1), 4), 'r-')
plot(meanRMSEs(2:size(meanRMSEs, 1), 1), meanRMSEs(2:size(meanRMSEs, 1), 4+errors_returned), 'b-')
legend('W angles', 'W/o angles')
xlabel('Number of hidden nodes')
ylabel('Mean error')
axis(axis_range)
title('Predict next three steps')
subplot(1, 3, 3)
hold on; box on
plot(meanRMSEs(2:size(meanRMSEs, 1), 1), meanRMSEs(2:size(meanRMSEs, 1), 3), 'r-')
plot(meanRMSEs(2:size(meanRMSEs, 1), 1), meanRMSEs(2:size(meanRMSEs, 1), 3+errors_returned), 'b-')
legend('W angles', 'W/o angles')
xlabel('Number of hidden nodes')
ylabel('Mean error')
axis(axis_range)
title('Predict next step')
