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
fig_legend_string = ["$\propto \vert \vert (V_k^T)_j \vert \vert_2^2$", ...
                     "$\propto l_{i, \lambda}(A), \lambda = \frac{\| A-\mathcal{T}_k(A) \|_F}{\sqrt{k}}$", ...
                      "$\sigma_{k+1}(A)$"];

%% compute projection errors

% sample proportionally to the rows norms of V_j
orthogonal_errors_row = zeros(num_avg, ranks);
for i=1:num_avg % average over multiple runs
    for j=1:ranks  % ranks from 1, ..., 50
        V_j = V(:, 1:j);  % take first k columns of V
        row_scores = zeros(n, 1); % compute scores
        for l=1:n
            row_scores(l) = norm(V_j(l, :))^2;
        end
        [~, projection_error] = RCCS(A, row_scores / j, j); % compute RCCS
        orthogonal_errors_row(i, j) = projection_error; % save error
    end
end
mean_errors_row = mean(orthogonal_errors_row);  % mean over num_avg runs

% ridge leverage scores with adaptive lambda
projection_errors_ridge = zeros(num_avg, ranks);
for i=1:num_avg % average over multiple runs
    for j=1:ranks % ranks from 1, ..., 50
        lambda = 1/sqrt(j) * sqrt(sum(gold_standards(j+1:end).^2)); % adaptive \lambda
        ridge_scores = diag(A' * pinv(A*A' + lambda^2 * eye(n)) * A); % compute scores
        [~, projection_error] = RCCS(A, ridge_scores / sum(ridge_scores), j); % compute RCCS
        projection_errors_ridge(i, j) = projection_error; % save error
    end
end
mean_errors_ridge = mean(projection_errors_ridge); % mean over num_avg runs

%% plot results
fig = figure();
x = (1:ranks);
semilogy(x, mean_errors_row, 'LineWidth', 2.5);
hold on
semilogy(x, mean_errors_ridge, 'LineWidth', 2.5);
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
legend('Location', 'northeast', 'FontSize', 15, 'NumColumns', 1);
saveas(fig, "../figures/lambda_tuning", "epsc");