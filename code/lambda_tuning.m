%% clean
clear
close all
clc

%% define Hilbert matrix and useful quantities
n = 100;
A = hilb(n);
num_avg = 100;
[U, S, V] = svd(A);
gold_standards = diag(S);
ranks = 50;

%% sample proportionally to the rows of V_k
errors_row = zeros(num_avg, ranks);  % initialize error vector

for i=1:num_avg
    for j=1:ranks
        V_j = V(:, 1:j);
        row_scores = zeros(n, 1);
        for l=1:n
            row_scores(l) = norm(V_j(l, :))^2 / j;
        end
        errors_row(i, j) = RCCS(A, row_scores / sum(row_scores), j);
    end
end

mean_errors_row = mean(errors_row);

%% cycle over lambda

% plot results
fig_legend_string = ["Rows of $V_k$", "$\sigma_{k+1}(A)$", "Ridge with adaptive $\lambda$"];

%% ridge leverage scores
errors_ridge = zeros(num_avg, ranks);  % initialize errors vector

% average over multiple runs
for i=1:num_avg
    % ranks from 1, ..., 50
    for j=1:ranks
        ridge_scores = zeros(n, 1);
        lambda = 1/j * sqrt(sum(gold_standards(j+1:end).^2));
        for k=1:n
            ridge_scores(k) = V(k, :) * diag(diag(S).^2 ./ (diag(S).^2 + lambda^2)) * V(k, :)';
        end
        ridge_scores = ridge_scores / sum(ridge_scores);
        errors_ridge(i, j) = RCCS(A, ridge_scores, j);  % compute error in 2-norm
    end
end

mean_errors_ridge = mean(errors_ridge);

%% plot results
fig = figure();
x = (1:ranks);
semilogy(x, mean_errors_row, 'LineWidth', 2.5);
hold on
semilogy(x, gold_standards(2:ranks+1), 'LineWidth', 2.0);
hold on
semilogy(x, mean_errors_ridge, 'LineWidth', 2.5);
xlabel("k", 'FontSize', 12);
ylabel("$\vert \vert A - C C^\dagger A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Low-rank approximation by column sampling", 'FontSize', 12);
legend(fig_legend_string, 'interpreter', 'latex');
legend('Location', 'best', 'FontSize', 12, 'NumColumns', 2);
saveas(fig, "../figures/lambda_tuning", "epsc");
saveas(fig, "../figures/lambda_tuning", "png");