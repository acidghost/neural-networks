clear all; close all; clc

fold_size = 5;

% Do line dataset
line = importdata('data/line.mat');
line_errors = ass3a_mlp(line, fold_size);
disp('MLP errors for line data')
disp(line_errors)
line_poly_errors = ass3a_poly(line, fold_size);
disp('Polyfit errors for line data')
disp(line_poly_errors)

% Do sin dataset
sinus = importdata('data/sinus.mat');
sin_errors = ass3a_mlp(sinus, fold_size);
disp('MLP errors for sinus data')
disp(sin_errors)
sin_poly_errors = ass3a_poly(sinus, fold_size);
disp('Polyfit errors for sinus data')
disp(sin_poly_errors)

% Do irregular dataset
irregular = importdata('data/irregular.mat');
irregular_errors = ass3a_mlp(irregular, fold_size);
disp('MLP errors for irregular data')
disp(irregular_errors)
irregular_poly_errors = ass3a_poly(irregular, fold_size);
disp('Polyfit errors for irregular data')
disp(irregular_poly_errors)
