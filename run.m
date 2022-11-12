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

%% leverage scores
errors_leverage = zeros(num_avg, ranks);  % initialize errors vector
errors_leverage_pseudo= zeros(num_avg, ranks);

% initialize scores
leverage_scores = zeros(n, 1);
for k=1:n
   leverage_scores(k) = A(:, k)' * pinv(A*A') * A(:, k);
end
leverage_scores = leverage_scores / sum(leverage_scores);

% average over multiple runs
for i=1:num_avg
    % ranks from 1, ..., 50
    for j=1:ranks
       C = zeros(n, j);
       for l=1:j
           p = sampling(leverage_scores);  % sample from distribution
           C(:, l) = A(:, p) / sqrt(j * leverage_scores(p));  % take column and rescale to have unbiased estimate
       end
       [Uc, ~, ~] = svd(C, "econ");  % economy SVD of C
       Q1 = Uc(:, 1:j);  % take left principal subspace
       errors_leverage(i, j) = norm(A-Q1*Q1'*A, 2);  % compute error in 2-norm
       errors_leverage_pseudo(i, j) = norm(A-C*pinv(C)*A, 2);
    end
end

mean_errors_leverage = mean(errors_leverage);
mean_errors_leverage_pseudo = mean(errors_leverage_pseudo);
std_errors_leverage = std(errors_leverage) / sqrt(num_avg);
std_errors_leverage_pseudo = std(errors_leverage_pseudo) / sqrt(num_avg);

%% ridge leverage scores
errors_ridge = zeros(num_avg, ranks);  % initialize errors vector
errors_ridge_pseudo= zeros(num_avg, ranks);

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
       errors_ridge_pseudo(i, j) = norm(A-C*pinv(C)*A, 2);
    end
end

mean_errors_ridge = mean(errors_ridge);
mean_errors_ridge_pseudo = mean(errors_ridge_pseudo);
std_errors_ridge = std(errors_ridge) / sqrt(num_avg);
std_errors_ridge_pseudo = std(errors_ridge_pseudo) / sqrt(num_avg);

%% uniform sampling
errors_uniform = zeros(num_avg, ranks);  % initialize errors vector
errors_uniform_pseudo = zeros(num_avg, ranks);  

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
       errors_uniform_pseudo(i, j) = norm(A-C*pinv(C)*A, 2);
    end
end

mean_errors_uniform = mean(errors_uniform);
mean_errors_uniform_pseudo = mean(errors_uniform_pseudo);
std_errors_uniform = std(errors_uniform) / sqrt(num_avg);
std_errors_uniform_pseudo = std(errors_uniform_pseudo) / sqrt(num_avg);


%% sample proportionally to the norm of the columns of A
errors_columns = zeros(num_avg, ranks);  % initialize errors vector
errors_columns_pseudo = zeros(num_avg, ranks);

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
       errors_columns_pseudo(i, j) = norm(A-C*pinv(C)*A, 2);
    end
end

mean_errors_columns = mean(errors_columns);
mean_errors_columns_pseudo = mean(errors_columns_pseudo);
std_errors_columns = std(errors_columns) / sqrt(num_avg);
std_errors_columns_pseudo = std(errors_columns_pseudo) / sqrt(num_avg);

%% sample proportionally to the rows of V_k
errors_row = zeros(num_avg, ranks);  % initialize error vector
errors_row_pseudo = zeros(num_avg, ranks);

for i=1:num_avg
    for k=1:ranks
        V_k = V(:, 1:k);
        row_scores = zeros(n, 1);
        for l=1:n
            row_scores(l) = norm(V_k(l, :))^2 / k;
        end
        C = zeros(n, k);
        for j=1:k
            p = sampling(row_scores);
            C(:, j) = A(:, p) / sqrt(k*row_scores(p));
        end
        [U, Sigma, V_hat] = svd(C, "econ");
        Q1 = U(:, 1:j);
        errors_row(i, j) = norm(A-Q1*Q1'*A, 2);
        errors_row_pseudo(i, j) = norm(A-C*pinv(C)*A, 2);
    end
end

mean_errors_row = mean(errors_row);
mean_errors_row_pseudo = mean(errors_row_pseudo);
std_errors_row = std(errors_row) / sqrt(num_avg);
std_errors_columns_row = std(errors_row_pseudo) / sqrt(num_avg);

%% plot results
fig = figure();
x = (1:ranks);
semilogy(x, mean_errors_ridge, 'LineWidth', 2.5);
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
ylabel("$\vert \vert A - Q Q^T A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Low-rank approximation by column sampling", 'FontSize', 12);
legend("Ridge leverage scores", "Columns norm sampling", "Uniform sampling", "Rows of $V_k$", "Leverage scores", "$\sigma_{k+1}(A)$", 'interpreter', 'latex');
legend('Location', 'best', 'FontSize', 12, 'NumColumns', 1);
saveas(fig, "figures/plot", "epsc");
saveas(fig, "figures/plot", "png");

fig2 = figure();
x = (1:ranks);
semilogy(x, mean_errors_ridge_pseudo, 'LineWidth', 2.5);
hold on
semilogy(x, mean_errors_columns_pseudo, 'LineWidth', 2.5);
hold on
semilogy(x, mean_errors_uniform_pseudo, 'LineWidth', 2.5);
hold on
semilogy(x, mean_errors_row_pseudo, 'LineWidth', 2.5);
hold on
semilogy(x, mean_errors_leverage_pseudo, 'LineWidth', 2.5);
hold on
semilogy(x, gold_standards(2:ranks+1), 'LineWidth', 2.0);
xlabel("k", 'FontSize', 12);
ylabel("$\vert \vert A - C C^\dagger A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Low-rank approximation by column sampling - Lambda ", 'FontSize', 12);
legend("Ridge leverage scores", "Columns norm sampling", "Uniform sampling", "Rows of $V_k$", "Leverage scores", "$\sigma_{k+1}(A)$", 'interpreter', 'latex');
legend('Location', 'best', 'FontSize', 12, 'NumColumns', 1);
saveas(fig2, "figures/plot2_lambda", "epsc");
saveas(fig2, "figures/plot2_lambda", "png");