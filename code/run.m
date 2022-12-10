%% clean
clear
close all
clc

%% import utilities and fix seed
addpath("utils")
rng("default")

%% run comparison between uniform sampling, columns norm sampling and ridge scores
tic
comparison
toc

%% run comparison between rows of Vk norm sampling and ridge scores with adaptive lambda
tic
lambda_tuning
toc

%% run repeated halving experiment
tic
repeated_halving
toc