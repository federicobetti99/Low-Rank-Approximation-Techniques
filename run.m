%% clean
clear
close all
clc

%% define Hilbert matrix
n = 100;
A = hilb(n);

%% useful quantities
num_avg = 100;
lambda = 1e-4;
[U, S, V] = svd(A);
gold_standards = diag(S);
ranks = 50;

%% compute sampling probabilities

% initialize classical leverage scores
leverage_scores = zeros(n, 1);
for k=1:n
   leverage_scores(k) = A(:, k)' * pinv(A*A') * A(:, k);
end
leverage_scores = leverage_scores / sum(leverage_scores);

% initialize ridge leverage scores (true, not estimates from random subset)
ridge_scores = zeros(n, 1);
for k=1:n
   ridge_scores(k) = V(k, :) * diag(diag(S).^2 ./ (diag(S).^2 + lambda^2)) * V(k, :)';
end
ridge_scores = ridge_scores / sum(ridge_scores);

% initialize upper estimates for ridge leverage scores
range = randsample(n, n/2);
M = A(:, range);
estimated_ridge_scores = zeros(n, 1);
for k=1:n
   estimated_ridge_scores(k) = A(:, k)' * pinv(M * M' + lambda.^2 * eye(n)) * A(:, k);
end
estimated_ridge_scores = estimated_ridge_scores / sum(estimated_ridge_scores);

% initialize uniform scores
uniform_scores = 1/n * ones(n, 1);

% initialize column scores
columns_scores = zeros(n, 1);
for k=1:n
   column_scores(k) = (norm(A(:, k)) / norm(A, "fro"))^2;
end

%% true classical leverage scores
% initialize errors vectors for all sampling probabilities
errors_leverage = zeros(num_avg, ranks);
errors_ridge = zeros(num_avg, ranks);
errors_ridge_estimates = zeros(num_avg, ranks); 
errors_uniform = zeros(num_avg, ranks);
errors_columns = zeros(num_avg, ranks);
errors_row = zeros(num_avg, ranks);

% average over multiple runs
for i=1:num_avg
    % ranks from 1, ..., 50
    for j=1:ranks
        % compute error in 2-norm
        errors_leverage(i, j) = RCCS(A, leverage_scores, j); 
        errors_ridge(i, j) = RCCS(A, ridge_scores, j);
        errors_ridge_estimates(i, j) = RCCS(A, estimated_ridge_scores, j);
        errors_uniform(i, j) = RCCS(A, uniform_scores, j);
        errors_columns(i, j) = RCCS(A, column_scores, j);
        V_j = V(:, 1:j);
        row_scores = zeros(n, 1);
        for l=1:n
            row_scores(l) = norm(V_j(l, :))^2 / k;
        end
        errors_row(i, j) = RCCS(A, row_scores / sum(row_scores), j);
    end
end

mean_errors_leverage = mean(errors_leverage);
mean_errors_ridge = mean(errors_ridge);
mean_errors_ridge_estimates = mean(errors_ridge_estimates);
mean_errors_uniform = mean(errors_uniform);
mean_errors_columns = mean(errors_columns);
mean_errors_row = mean(errors_row);

%% plot results
fig = figure();
x = (1:ranks);
semilogy(x, mean_errors_ridge, 'LineWidth', 2.5);
hold on
semilogy(x, mean_errors_ridge_estimates, 'LineWidth', 2.5);
hold on
semilogy(x, mean_errors_columns, 'LineWidth', 2.5);
hold on
semilogy(x, mean_errors_uniform, 'LineWidth', 2.5);
hold on
semilogy(x, mean_errors_row, 'LineWidth', 2.5);
hold on
semilogy(x, mean_errors_leverage, 'LineWidth', 2.5);
hold on
semilogy(x, gold_standards(2:ranks+1), 'LineWidth', 2.0);
xlabel("k", 'FontSize', 12);
ylabel("$\vert \vert A - C C^\dagger A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Low-rank approximation by column sampling", 'FontSize', 12);
legend("$\sim l_{i, \lambda}(A)$", "$\sim l_{i, \lambda}^{M}(A)$", "$\sim \vert \vert a_j \vert \vert_2^2$", ...
    "$\sim 1/n$", "$\sim \vert \vert V_k(j, :) \vert \vert_2^2$", ...
    "$\sim l_i(A)$", "$\sigma_{k+1}(A)$", 'interpreter', 'latex'); 
legend('Location', 'southoutside', 'FontSize', 18, 'NumColumns', 3);
saveas(fig, "plot", "png");