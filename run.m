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


%% ridge leverage scores
errors_ridge = zeros(num_avg, ranks);  % initialize errors vector

% initialize scores
ridge_scores = zeros(n, 1);
for k=1:n
   ridge_scores(k) = V(k, :) * diag(diag(S).^2 ./ (diag(S).^2 + lambda^2)) * V(k, :)';
end
ridge_scores = ridge_scores / sum(ridge_scores);

% average over multiple runs
for i=1:num_avg
    % ranks from 1, ..., 50
    for j=1:ranks
       C = zeros(n, j);
       for l=1:j
           p = sampling(ridge_scores);  % sample from distribution
           C(:, l) = A(:, p) / sqrt(j * ridge_scores(p));  % take column and rescale to have unbiased estimate
       end
       [Uc, ~, ~] = svd(C, "econ");  % economy SVD of C
       Q1 = Uc(:, 1:j);  % take left principal subspace
       errors_ridge(i, j) = norm(A-Q1*Q1'*A, 2);  % compute error in 2-norm
    end
end

mean_errors_ridge = mean(errors_ridge);
std_errors_ridge = std(errors_ridge) / sqrt(num_avg);


%% uniform sampling
errors_uniform = zeros(num_avg, ranks);  % initialize errors vector

% initialize scores
uniform_scores = 1/n * ones(n, 1);

% average over multiple runs
for i=1:num_avg
    % ranks from 1, ..., 50
    for j=1:ranks
       C = zeros(n, 1);
       for l=1:j
           p = sampling(uniform_scores);  % sample from distribution 
           C(:, l) = A(:, p) / sqrt(j * uniform_scores(p));  % take column and rescale to have unbiased estimate
       end
       [Uc, ~, ~] = svd(C, "econ");  % economy SVD of C
       Q1 = Uc(:, 1:j);  % take left principal subspace
       errors_uniform(i, j) = norm(A-Q1*Q1'*A, 2);  % compute error in 2-norm
    end
end

mean_errors_uniform = mean(errors_uniform);
std_errors_uniform = std(errors_uniform);


%% sample proportionally to the norm of the columns of A
errors_columns = zeros(num_avg, ranks);  % initialize errors vector

% initialize scores
columns_scores = zeros(n, 1);
for k=1:n
   column_scores(k) = (norm(A(:, k)) / norm(A, "fro"))^2;
end

% average over multiple runs
for i=1:num_avg
    % ranks from 1, ..., 50
    for j=1:ranks
       C = zeros(n, j);
       for l=1:j
           p = sampling(column_scores);  % sample from distribution
           C(:, l) = A(:, p) / sqrt(j * column_scores(p));  % take column and rescale to have unbiased estimate
       end
       [Uc, ~, ~] = svd(C, "econ");  % economy SVD of C
       Q1 = Uc(:, 1:j);  % take left principal subspace
       errors_columns(i, j) = norm(A-Q1*Q1'*A, 2);  % compute error in 2-norm
    end
end

mean_errors_columns = mean(errors_columns);
std_errors_columns = std(errors_columns);


%% plot results
fig = figure();
x = (1:ranks);
semilogy(x, mean_errors_ridge, 'LineWidth', 2.5);
hold on
semilogy(x, mean_errors_columns, 'LineWidth', 2.5);
hold on
semilogy(x, mean_errors_uniform, 'LineWidth', 2.5);
hold on
semilogy(x, gold_standards(2:ranks+1), 'LineWidth', 2.0);
xlabel("k", 'FontSize', 12);
ylabel("$\vert \vert A - Q Q^T A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Low-rank approximation by column sampling", 'FontSize', 12);
legend("Ridge leverage scores", "Columns norm sampling", "Uniform sampling", "$\sigma_{k+1}(A)$", 'interpreter', 'latex');
legend('Location', 'best', 'FontSize', 12, 'NumColumns', 1);
saveas(fig, "plot", "epsc");