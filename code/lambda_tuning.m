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
[U, S, V] = svd(A);
gold_standards = diag(S);
ranks = 50;

%% sample proportionally to the rows of V_k
orthogonal_errors_row = zeros(num_avg, ranks);
oblique_errors_row = zeros(num_avg, ranks);

for i=1:num_avg
    for j=1:ranks
        V_j = V(:, 1:j);
        row_scores = zeros(n, 1);
        for l=1:n
            row_scores(l) = norm(V_j(l, :))^2 / j;
        end
        [orthogonal_error, oblique_error] = RCCS(A, row_scores / sum(row_scores), j); 
        orthogonal_errors_row(i, j) = orthogonal_error;
        oblique_errors_row(i, j) = oblique_error;
    end
end

orthogonal_mean_errors_row = mean(orthogonal_errors_row);
oblique_mean_errors_row = mean(oblique_errors_row);

%% cycle over lambda

% plot results
fig_legend_string = ["$\sim \vert \vert (V_k^T)_j \vert \vert_2^2$", "$\sigma_{k+1}(A)$", "$l_{i, \lambda}(A)$ with adaptive $\lambda$"];

%% ridge leverage scores
orthogonal_errors_ridge = zeros(num_avg, ranks);
oblique_errors_ridge = zeros(num_avg, ranks);

% average over multiple runs
for i=1:num_avg
    % ranks from 1, ..., 50
    for j=1:ranks
        ridge_scores = zeros(n, 1);
        lambda = 1/sqrt(j) * sqrt(sum(gold_standards(j+1:end).^2));
        for k=1:n
            ridge_scores(k) = V(k, :) * diag(diag(S).^2 ./ (diag(S).^2 + lambda^2)) * V(k, :)';
        end
        ridge_scores = ridge_scores / sum(ridge_scores);
        [orthogonal_error, oblique_error] = RCCS(A, ridge_scores, j);
        orthogonal_errors_ridge(i, j) = orthogonal_error;
        oblique_errors_ridge(i, j) = oblique_error;
    end
end

orthogonal_mean_errors_ridge = mean(orthogonal_errors_ridge);
oblique_mean_errors_ridge = mean(oblique_errors_ridge);

%% plot results for orthogonal projection errors
fig = figure();
x = (1:ranks);
semilogy(x, orthogonal_mean_errors_row, 'LineWidth', 2.5);
hold on
semilogy(x, gold_standards(2:ranks+1), 'LineWidth', 2.0);
hold on
semilogy(x, orthogonal_mean_errors_ridge, 'LineWidth', 2.5);

xlabel("k", 'FontSize', 12);
ylabel("$\vert \vert A - Q Q^T A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Low-rank approximation by column sampling", 'FontSize', 12);
legend(fig_legend_string, 'interpreter', 'latex');
legend('Location', 'southwest', 'FontSize', 15, 'NumColumns', 2);
saveas(fig, "../figures/orthogonal_lambda_tuning", "epsc");

%% plot results for orthogonal projection errors
fig2 = figure();
x = (1:ranks);
semilogy(x, oblique_mean_errors_row, 'LineWidth', 2.5);
hold on
semilogy(x, gold_standards(2:ranks+1), 'LineWidth', 2.0);
hold on
semilogy(x, oblique_mean_errors_ridge, 'LineWidth', 2.5);

xlabel("k", 'FontSize', 12);
ylabel("$\vert \vert A - C C^\dagger A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Low-rank approximation by column sampling", 'FontSize', 12);
legend(fig_legend_string, 'interpreter', 'latex');
legend('Location', 'southwest', 'FontSize', 15, 'NumColumns', 2);
saveas(fig2, "../figures/oblique_lambda_tuning", "epsc");