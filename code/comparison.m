%% clean
clear
close all
clc

%% import utilities and fix seed
addpath("utils")
rng("default")
rng(0)

%% define Hilbert matrix and useful quantities
n = 100;
A = hilb(n);
num_avg = 20;
[U, S, V] = svd(A);
gold_standards = diag(S);
ranks = 50;
fig_legend_string = ["$\propto l_{i, \lambda}(A)$", "$\propto 1/n$", ...
                     "$\propto \| a_j \|_2^2$", "$\sigma_{k+1}(A)$"];

%% compute projection errors

% ridge leverage scores
lambda = 1e-4; % regularization parameter
ridge_scores = diag(V * diag(diag(S).^2 ./ (diag(S).^2 + lambda^2)) * V'); % compute ridge scores
mean_errors_ridge = compute_averages(A, ridge_scores / sum(ridge_scores), num_avg, ranks); % mean over num_avg runs

% uniform scores
uniform_scores = 1/n * ones(n, 1); % compute uniform scores
mean_errors_uniform = compute_averages(A, uniform_scores, num_avg, ranks); % mean over num_avg runs

% columns norm scores
column_scores = sum(A.^2, 2) / norm(A, "fro")^2; % compute columns norm scores
mean_errors_columns = compute_averages(A, column_scores, num_avg, ranks); % mean over num_avg runs

%% plot results
fig = figure();
x = (1:ranks);
semilogy(x, mean_errors_ridge, 'LineWidth', 2.5);
hold on
semilogy(x, mean_errors_uniform, 'LineWidth', 2.5);
hold on
semilogy(x, mean_errors_columns, 'LineWidth', 2.5);
hold on
semilogy(x, gold_standards(2:ranks+1), 'LineWidth', 2.0);
hold on
xlabel("k", 'FontSize', 12);
ylabel("$\vert \vert A - Q Q^T A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Low-rank approximation by column sampling", 'FontSize', 12);
legend(fig_legend_string, 'interpreter', 'latex');
legend('Location', 'northeast', 'FontSize', 15, 'NumColumns', 3);
saveas(fig, "../figures/plot", "epsc");