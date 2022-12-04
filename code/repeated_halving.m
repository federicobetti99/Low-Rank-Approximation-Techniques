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
lambda = 1e-4;

%% cycle over lambda

% plot results
fig_legend_string = ["$\sigma_{k+1}(A)$", "$l_{i, \lambda}(A)$", "$l_{i, \lambda}^M(A)$"];

%% ridge leverage scores

projection_errors_ridge = zeros(num_avg, ranks);

% average over multiple runs
for i=1:num_avg
    % ranks from 1, ..., 50
    for j=1:ranks
        ridge_scores = zeros(n, 1);
        lambda = 1e-4;
        for k=1:n
            ridge_scores(k) = V(k, :) * diag(diag(S).^2 ./ (diag(S).^2 + lambda^2)) * V(k, :)';
        end
        ridge_scores = ridge_scores / sum(ridge_scores);
        [~, orthogonal_error] = RCCS(A, ridge_scores, j);
        projection_errors_ridge(i, j) = orthogonal_error;
    end
end

mean_errors_ridge = mean(projection_errors_ridge);

%% Estimates with repeated halving procedure

projection_errors_estimate_ridge = zeros(num_avg, ranks);

% average over multiple runs
for i=1:num_avg
           
    Ah = A;
    col = n;
    M = zeros(n, n);
    
    % ranks from 1, ..., 50
    for j=1:ranks
       while col > j*log(j) && col > 1
            col2 = fix(col/2);
            M = Ah(:, randsample(col, col2)); 
            Ah = M;
            col = col2;
       end
       [~, Sigma_M, ~] = svd(M, "econ");
       gold_standard_M = diag(Sigma_M);
       lambda = 1/sqrt(j) * sqrt(sum(gold_standard_M(j+1:end).^2));
       estimated_ridge_scores = diag(A'*pinv(M*M' + lambda.^2*eye(n))*A);
       [~, projection_error] = RCCS(A, estimated_ridge_scores / sum(estimated_ridge_scores), j);
       projection_errors_estimate_ridge(i, j) = projection_error;
    end
end

mean_errors_estimate_ridge = mean(projection_errors_estimate_ridge);

%% plot results for orthogonal projection errors
fig = figure();
x = (1:ranks);
semilogy(x, gold_standards(2:ranks+1), 'LineWidth', 2.0);
hold on
semilogy(x, mean_errors_ridge, 'LineWidth', 2.5);
hold on
semilogy(x, mean_errors_estimate_ridge, 'LineWidth', 2.5); %

xlabel("k", 'FontSize', 12);
ylabel("$\vert \vert A - Q Q^T A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Low-rank approximation by column sampling", 'FontSize', 12);
legend(fig_legend_string, 'interpreter', 'latex');
legend('Location', 'southwest', 'FontSize', 15, 'NumColumns', 2);
saveas(fig, "../figures/repeated_halving", "epsc");