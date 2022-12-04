%% clean
clear
close all
clc

%% import utilities
addpath("utils")

%% define Hilbert matrix and useful quantities
n = 100;
A = hilb(n);
num_avg = 100;
lambda = 1e-4;
[U, S, V] = svd(A);
gold_standards = diag(S);
ranks = 50;
fig_legend_string = "$\sigma_{k+1}(A)$";

%% choose sampling probabilities to compare
show_ridge_scores = 1;
show_uniform_scores = 1;
show_column_scores = 1;

%% compute averages for selected sampling probabilities

if (show_ridge_scores) % ridge leverage scores
    ridge_scores = zeros(n, 1);
    for k=1:n
       ridge_scores(k) = V(k, :) * diag(diag(S).^2 ./ (diag(S).^2 + lambda^2)) * V(k, :)';
    end
    mean_errors_ridge = compute_averages(A, ridge_scores / sum(ridge_scores), num_avg, ranks);
    fig_legend_string = [fig_legend_string, "$\sim l_{i, \lambda}(A)$"];
end

if (show_uniform_scores) % uniform sampling over the columns
    uniform_scores = 1/n * ones(n, 1);
    mean_errors_uniform = compute_averages(A, uniform_scores, num_avg, ranks);
    fig_legend_string = [fig_legend_string, "$\sim 1/n$"];
end

if (show_column_scores) % sample proportionally to the columns norm of A
    column_scores = sum(A.^2, 2);
    mean_errors_columns = compute_averages(A, column_scores / sum(column_scores), num_avg, ranks);
    fig_legend_string = [fig_legend_string, "$\sim \vert \vert a_j \vert \vert_2^2$"];
end

%% plot results with orthogonal projection error
fig = figure();
x = (1:ranks);
semilogy(x, gold_standards(2:ranks+1), 'LineWidth', 2.0);
hold on
if (show_ridge_scores)
    semilogy(x, mean_errors_ridge, 'LineWidth', 2.5);
    hold on
end
if (show_uniform_scores)
    semilogy(x, mean_errors_uniform, 'LineWidth', 2.5);
    hold on
end
if (show_column_scores)
    semilogy(x, mean_errors_columns, 'LineWidth', 2.5);
    hold on
end
xlabel("k", 'FontSize', 12);
ylabel("$\vert \vert A - Q Q^T A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Low-rank approximation by column sampling", 'FontSize', 12);
legend(fig_legend_string, 'interpreter', 'latex');
legend('Location', 'southwest', 'FontSize', 15, 'NumColumns', 3);
saveas(fig, "../figures/plot", "epsc");