%% clean
clear
close all
clc

%% define Hilbert matrix and useful quantities
n = 100;
A = hilb(n);
num_avg = 100;
lambda = 1e-4;
[U, S, V] = svd(A);
gold_standards = diag(S);
ranks = 50;

fig = figure();
x = (1:ranks);

fig_legend_string = "$\sigma_{k+1}(A)$";
semilogy(x, gold_standards(2:ranks+1), 'LineWidth', 2.0);
hold on

%% classical leverage scores

% initialize classical leverage scores
leverage_scores = zeros(n, 1);
for k=1:n
   leverage_scores(k) = A(:, k)' * pinv(A*A') * A(:, k);
end

errors_leverage_scores = compute_averages(A, leverage_scores / sum(leverage_scores), num_avg, ranks);
fig_legend_string = [fig_legend_string, "$\sim l_i(A)$"];
semilogy(x, errors_leverage_scores, 'LineWidth', 2.5);
hold on

%% exact ridge scores

% initialize ridge leverage scores (true, not estimates from random subset)
ridge_scores = zeros(n, 1);
for k=1:n
   ridge_scores(k) = V(k, :) * diag(diag(S).^2 ./ (diag(S).^2 + lambda^2)) * V(k, :)';
end

errors_ridge_scores = compute_averages(A, ridge_scores / sum(ridge_scores), num_avg, ranks);
fig_legend_string = [fig_legend_string, "$\sim l_{i, \lambda}(A)$"];
semilogy(x, errors_ridge_scores, 'LineWidth', 2.5);
hold on

%%  overestimates for ridge scores

% initialize upper estimates for ridge leverage scores
range = randsample(n, n/2);
M = A(:, range);
estimated_ridge_scores = zeros(n, 1);
for k=1:n
   estimated_ridge_scores(k) = A(:, k)' * pinv(M*M' + lambda.^2*eye(n)) * A(:, k);
end

errors_estimated_ridge_scores = compute_averages(A, estimated_ridge_scores / sum(estimated_ridge_scores), num_avg, ranks);
fig_legend_string = [fig_legend_string, "$\sim l_{i, \lambda}^{M}(A)$"];
semilogy(x, errors_estimated_ridge_scores, 'LineWidth', 2.5);
hold on

%% uniform sampling over the columns

% initialize uniform scores
uniform_scores = 1/n * ones(n, 1);

errors_uniform_scores = compute_averages(A, uniform_scores, num_avg, ranks);
fig_legend_string = [fig_legend_string, "$\sim 1/n$"];
semilogy(x, errors_uniform_scores, 'LineWidth', 2.5);
hold on

%% sample proportionally to the columns norm of A

% initialize column scores
columns_scores = zeros(n, 1);
for k=1:n
   column_scores(k) = (norm(A(:, k)) / norm(A, "fro"))^2;
end

errors_columns_scores = compute_averages(A, column_scores, num_avg, ranks);
fig_legend_string = [fig_legend_string, "$\sim \vert \vert a_j \vert \vert_2^2$"];
semilogy(x, errors_columns_scores, 'LineWidth', 2.5);
hold on

%% plot results
xlabel("k", 'FontSize', 12);
ylabel("$\vert \vert A - C C^\dagger A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Low-rank approximation by column sampling", 'FontSize', 12);
legend(fig_legend_string, 'interpreter', 'latex');
legend('Location', 'best', 'FontSize', 15, 'NumColumns', 3);
saveas(fig, "../figures/plot", "png");
saveas(fig, "../figures/plot", "epsc");