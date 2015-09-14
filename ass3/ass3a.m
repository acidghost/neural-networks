irregular = importdata('data/irregular.mat');
line = importdata('data/line.mat');
sinus = importdata('data/sinus.mat');

subplot(1, 3, 1)
gscatter(irregular.x, irregular.t)
title('Irregular')

subplot(1, 3, 2)
gscatter(line.x, line.t)
title('Line')

subplot(1, 3, 3)
gscatter(sinus.x, sinus.t)
title('Sinus')
