function net = rbfsetbf(net, options, x, initialization)
%RBFSETBF Set basis functions of RBF from data.
%
%	Description
%	NET = RBFSETBF(NET, OPTIONS, X) sets the basis functions of the RBF
%	network NET so that they model the unconditional density of the
%	dataset X.  This is done by training a GMM with spherical covariances
%	using GMMEM.  The OPTIONS vector is passed to GMMEM. The widths of
%	the functions are set by a call to RBFSETFW.
%
%	See also
%	RBFTRAIN, RBFSETFW, GMMEM
%

%	Copyright (c) Ian T Nabney (1996-2001)

errstring = consist(net, 'rbf', x);
if ~isempty(errstring)
  error(errstring);
end

if nargin == 3
    initialization = 'random';
end

% Initialise the parameters from the input data
% Just use a small number of k means iterations
kmoptions = zeros(1, 18);
kmoptions(1) = -1;	% Turn off warnings
kmoptions(14) = 50;  % 50 iterations should do the trick

% Try a simple variant of k-means clustering....

clusters = net.nhidden;
if strncmp(initialization, '+', 1) == 1
    % Initialize clusters: Using Kmeans++
    centres = nan(clusters, size(x, 2));
    samples = size(x, 1);
    disp(['Starting K-means with ', num2str(clusters), ' clusters and ', num2str(samples), ' samples']);
    % Step1: Choose one center uniformly at random from among the data points.
    r = round((rand * (samples-1)) + 1);
    centres(1, :) = x(r, :);
    for k = 2:clusters;
        % Step2: For each data point x, compute D(x), the distance between x 
        % and the nearest center that has already been chosen.
        distances = zeros(samples, 1);
        for i = 1:samples;
            dis = pdist2(x(i, :), centres(1:k-1, :));
            distances(i) = min(dis);
        end
        % Step3: Choose one new data point at random as a new center, using 
        % a weighted probability distribution where a point x is chosen 
        % with probability proportional to D(x)^2.
        new_centre = RouletteWheelSelection(distances.^2);
        centres(k, :) = x(new_centre, :);
    end
else
    % rng(42, 'twister');
    centres = rand(clusters,size(x,2)); 
end

centres = kmeans(centres, x, kmoptions);

% Now set the centres of the RBF from the centres of the mixture model
net.c = centres;

% options(7) gives scale of function widths
net = rbfsetfw(net, options(7));

end

function [index] =  RouletteWheelSelection(arrayInput)

len = length(arrayInput);

% if input is one element then just return rightaway
if len ==1
    index =1;
    return;
end

if (~isempty(find(arrayInput<1, 1)))
    if (min(arrayInput) ~=0)
    arrayInput = 1/min(arrayInput)*arrayInput;
    else
    temp= arrayInput;
    temp(arrayInput==0) = inf;
    arrayInput = 1/min(temp)*arrayInput;
    end
end

temp = 0;
tempProb = zeros(1,len);

for i= 1:len
    tempProb(i) = temp + arrayInput(i);
    temp = tempProb(i);
end

i = fix(rand*floor(tempProb(end)))+1;
index = find(tempProb >= i, 1 );

end
