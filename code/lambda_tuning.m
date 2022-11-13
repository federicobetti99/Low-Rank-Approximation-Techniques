%% clean
clear
close all
clc

%% define Hilbert matrix
n = 100;
A = hilb(n);

%% useful quantities
num_avg = 100;
lambdas = 10.^-(0:5:10);
[U, S, V] = svd(A);
gold_standards = diag(S);
ranks = 50;

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


%% cycle over lambda

% plot results
fig_legend_string = ["Rows of $V_k$", "$\sigma_{k+1}(A)$"];
fig2_legend_string = ["Rows of $V_k$", "$\sigma_{k+1}(A)$"];

means_lambda = zeros(ranks, length(lambdas));
means_lambda_pseudo = zeros(ranks, length(lambdas));

for z=1:length(lambdas)
    lambda = lambdas(z);

    % ridge leverage scores
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
    
    means_lambdas(:, z) = mean(errors_ridge);
    means_lambdas_pseudo(:, z) = mean(errors_ridge_pseudo);
    fig_legend_string = [fig_legend_string, sprintf("$\\lambda = 10^{%d}$", log10(lambda))];
    fig2_legend_string = [fig2_legend_string, sprintf("$\\lambda = 10^{%d}$", log10(lambda))];   
end


%% plot results
fig = figure();
x = (1:ranks);
semilogy(x, mean_errors_row, 'LineWidth', 2.5);
hold on
semilogy(x, gold_standards(2:ranks+1), 'LineWidth', 2.0);
hold on
for z=1:length(lambdas)
    semilogy(x, means_lambdas(:, z), 'LineWidth', 2.0);
end
xlabel("k", 'FontSize', 12);
ylabel("$\vert \vert A - Q Q^T A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Low-rank approximation by column sampling", 'FontSize', 12);
legend(fig_legend_string, 'interpreter', 'latex');
legend('Location', 'best', 'FontSize', 12, 'NumColumns', 2);

fig2 = figure();
semilogy(x, mean_errors_row_pseudo, 'LineWidth', 2.5);
hold on
semilogy(x, gold_standards(2:ranks+1), 'LineWidth', 2.0);
hold on
for z=1:length(lambdas)
    semilogy(x, means_lambdas_pseudo(:, z), 'LineWidth', 2.0);
end
xlabel("k", 'FontSize', 12);
ylabel("$\vert \vert A - C C^\dagger A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Low-rank approximation by column sampling", 'FontSize', 12);
legend(fig2_legend_string, 'interpreter', 'latex');
legend('Location', 'best', 'FontSize', 12, 'NumColumns', 2);

saveas(fig, "figures/lambda_tuning", "epsc");
saveas(fig, "figures/lambda_tuning", "png");
saveas(fig2, "figures/lambda_tuning2", "epsc");
saveas(fig2, "figures/lambda_tuning2", "png");