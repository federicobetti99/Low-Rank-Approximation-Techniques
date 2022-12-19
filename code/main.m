%% clean
clear
close all
clc

% Running the three subsections below reproduces the plots shown in
% the report in the same order as they appear in the latter

%% run comparison between uniform sampling, columns norm sampling and ridge scores
tic
comparison
toc

%% plot cumulative sum of sketching distributions
tic
dist_scores
toc

%% run comparison between rows of Vk norm sampling and ridge scores with adaptive lambda
tic
lambda_tuning
toc