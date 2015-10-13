% Spatial crowd behaviour simulator using a neural network

% Mark Hoogendoorn

% May 2012



%clear all;



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



% relevant points (e.g. corners of buildings)

% NOTE: "max_y" operation was added, because image needs to be y-mirrorred

points = [546.8, max_y-478.0;

          507.5, max_y-330.6;

          240.6, max_y-218.6;

          184.7, max_y-331.3];



% sources of panic (e.g. shouting individual)

source = [542.0, max_y-439.0];

      

% solid lines that cannot be passed (buildings, fences, ...)

% NOTE: some lines have been made longer than represented in the data

lines = [321.2, max_y-314.5, 240.0, 300.0;          % was 321.2, max_y-314.5, 286.1, max_y-396.9;

         321.2, max_y-314.5, 275.5, max_y-292.0;

         383.0, max_y-336.4, 342.0, 300.0;          % was 383.0, max_y-336.4, 365.2, max_y-407.2

         383.0, max_y-336.4, 600.0, 358.0;          % was 383.0, max_y-336.4, 431.2, max_y-359.6

         385.3, max_y-321.6, 448.0, max_y-347.4;

         448.0, max_y-347.4, 449.0, max_y-313.9;

         449.0, max_y-313.9, 390.5, max_y-292.0;

         390.5, max_y-292.0, 385.3, max_y-321.6];



%%%%%%%%%%%%%%%%%%%%%%

%%% Pre-processing %%%

%%%%%%%%%%%%%%%%%%%%%%


% calculate slopes and bases of solid lines

for l = 1:size(lines, 1)

    slopes(l) = (lines(l, 4) - lines(l, 2)) / (lines(l, 3) - lines(l, 1));

    bases(l) = lines(l, 2) - slopes(l) * lines(l, 1);

end



% place people in the environment


for i=1:end_time
% Do some drawing....
clf;
hold on;
grid on;
set(gcf, 'Position', [265 5 750 1000])
set(gca,'DataAspectRatio',[1 1 1]);
axis([0 max_x+1 0 max_y+1]);
    
  % draw lines

  for l = 1:size(lines, 1)
      line([lines(l,1) lines(l,3)],[lines(l,2) lines(l,4)],'Color',[0 0 0],'LineStyle','-');
  end

  % draw environmental objects (circle)

  radius = sqrt((386.9-374.3)^2+(208.9-257.2)^2);
  t=(0:50)*2*pi/50;
  x=radius*cos(t)+386.9;
  y=radius*sin(t)+max_y-208.9;
  plot(x,y,'Color',[0 0 0]);

  % draw relevant points

  for l = 1:size(points, 1)
        plot(points(l,1),points(l,2),'Color',[0 0 0],'Marker','.','MarkerSize',20);
  end

  % draw source
    
  plot(source(1),source(2),'Color',[1 0 0],'Marker','.','MarkerSize',20);
  
  for a = 1:number_of_agents
      % draw person
      plot(data(i,(2*a)-1),data(i,(2*a)),'Color',[0 0 1],'Marker','.','MarkerSize',10);
  end
  pause(0.01);
end
