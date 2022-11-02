%% clean
clear
close all
clc

%% define Hilbert matrix
n = 100;
A = hilb(n);

%% useful quantities
num_avg = 20;
lambda = 1e-4;
[U, S, V] = svd(A);
gold_standards = diag(S);
ranks = 50;


%% ridge leverage scoring
errors_ridge = zeros(num_avg, ranks);  % initialize errors vector
for i=1:num_avg
    for j=1:ranks
       ridge_scores = zeros(n, 1);
       for k=1:n
           ridge_scores(k) = V(k, :) * diag(diag(S).^2 ./ (diag(S).^2 + lambda^2)) * V(k, :)';
       end
       ridge_scores = ridge_scores / sum(ridge_scores);
       indeces = randsample((1:n), j, true, ridge_scores);
       C = zeros(n, j);
       for l=1:j
           C(:, l) = A(:, indeces(l)) / sqrt(j * ridge_scores(indeces(l)));
       end
       errors_ridge(i, j) = norm(A-C*pinv(C)*A, 2);
    end
end
mean_errors_ridge = mean(errors_ridge);
std_errors_ridge = std(errors_ridge) / sqrt(num_avg);

%% uniform sampling
errors_uniform = zeros(num_avg, ranks);  % initialize errors vector
B = A;  % copy of A for replacement
for i=1:num_avg
    for j=1:ranks
       indeces = randsample((1:n), j, true);
       B = A(:, indeces) ./ sqrt(j * 1/n);
       C = zeros(n, j);
       for l=1:j
           C(:, l) = A(:, indeces(l)) / sqrt(j * 1/n);
       end
       errors_uniform(i, j) = norm(A-C*pinv(C)*A, 2);
    end
end
mean_errors_uniform = mean(errors_uniform);
std_errors_uniform = std(errors_uniform);

%% proportionally to the rows of V
errors_columns = zeros(num_avg, ranks);  % initialize errors vector
for i=1:num_avg
    for j=1:ranks
       columns_scores = zeros(n, 1);
       for k=1:n
           column_scores(k) = norm(A(:, k), 2) / norm(A, "fro");
       end
       indeces = randsample((1:n), j, true, column_scores);
       C = zeros(n, j);
       for l=1:j
           C(:, l) = A(:, indeces(l)) / sqrt(j * column_scores(indeces(l)));
       end
       errors_columns(i, j) = norm(A-C*pinv(C)*A, 2);
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
xlabel("k", 'FontSize', 10);
hold on
semilogy(x, gold_standards(2:ranks+1), 'LineWidth', 1.5);
ylabel("$\vert \vert A - C C^\dagger A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 10);
title("Low-rank approximation by column sampling", 'FontSize', 15);
legend("Ridge leverage scores", "Columns norm sampling", "Uniform sampling", "$\sigma_{k+1}(A)$", 'interpreter', 'latex');
legend('Location', 'southeast', 'FontSize', 10, 'NumColumns', 1);
saveas(fig, "plot", "epsc");