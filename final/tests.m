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
% 
% Xpath = path(1:end_time-1, :);
% Xtra = [Xpath pdist2(Xpath, source)];
% Ytra = path(2:end_time, :);
% model = initlssvm( Xtra, Ytra, 'f', [], [], 'RBF_kernel' );
% model = tunelssvm(model, 'simplex', 'crossvalidatelssvm', { 10, 'mae' });
% model = trainlssvm(model);
% Yhs = simlssvm(model, Xtra);
% 
% error = diag(pdist2(Yhs, Ytra));
% disp(['Error is ', num2str(sum(error))])

Xpaths = zeros((end_time-1) * number_of_agents, 2);
Ypaths = zeros((end_time-1) * number_of_agents, 2);
Xcounter = 1;
Ycounter = 1;
for agent = 1:number_of_agents;
    for i = 1:end_time;
        if i == 1
            Xpaths(Xcounter, :) = [data(i, (2*a)-1), data(i, 2*a)];
            Xcounter = Xcounter + 1;
        elseif i == end_time;
            Ypaths(Ycounter, :) = [data(i, (2*a)-1), data(i, 2*a)];
            Ycounter = Ycounter + 1;
        else
            Xpaths(Xcounter, :) = [data(i, (2*a)-1), data(i, 2*a)];
            Xcounter = Xcounter + 1;
            Ypaths(Ycounter, :) = [data(i, (2*a)-1), data(i, 2*a)];
            Ycounter = Ycounter + 1;
        end
    end
end

Xtra = [Xpaths pdist2(Xpaths, source)];
model = initlssvm( Xtra, Ypaths, 'f', [], [], 'RBF_kernel' );
model = tunelssvm(model, 'simplex', 'crossvalidatelssvm', { 10, 'mae' });
model = trainlssvm(model);
Yhs = simlssvm(model, Xtra);

error = diag(pdist2(Yhs, Ypaths));
MSE = mean(sqrt(error));
disp(['MSE is ', num2str(MSE)])
