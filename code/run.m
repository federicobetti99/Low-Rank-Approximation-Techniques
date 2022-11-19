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
fig_legend_string = "$\sigma_{k+1}(A)$";

%% choose sampling probabilities to compare
show_leverage_scores = 0;
show_ridge_scores = 1;
show_estimated_ridge_scores = 0;
show_uniform_scores = 1;
show_column_scores = 1;

%% compute averages for selected sampling probabilities

if (show_leverage_scores) % classical leverage scores
    leverage_scores = diag(A'*pinv(A*A')*A);
    [orthogonal_errors_leverage_scores, oblique_errors_leverage_scores] = compute_averages(A, ...
        leverage_scores / sum(leverage_scores), num_avg, ranks);
    fig_legend_string = [fig_legend_string, "$\sim l_i(A)$"];
end

if (show_ridge_scores) % exact ridge scores
    ridge_scores = zeros(n, 1);
    for k=1:n
       ridge_scores(k) = V(k, :) * diag(diag(S).^2 ./ (diag(S).^2 + lambda^2)) * V(k, :)';
    end
    [orthogonal_errors_ridge_scores, oblique_errors_ridge_scores] = compute_averages(A, ...
        ridge_scores / sum(ridge_scores), num_avg, ranks);
    fig_legend_string = [fig_legend_string, "$\sim l_{i, \lambda}(A)$"];
end

if (show_estimated_ridge_scores) %  overestimates for ridge scores
    M = A(:, randsample(n, n/2));
    estimated_ridge_scores = diag(A'*pinv(M*M' + lambda.^2*eye(n))*A);
    [orthogonal_errors_estimated_ridge_scores, oblique_errors_estimated_ridge_scores] = compute_averages(A, ...
        estimated_ridge_scores / sum(estimated_ridge_scores), num_avg, ranks);
    fig_legend_string = [fig_legend_string, "$\sim l_{i, \lambda}^{M}(A)$"];
end

if (show_uniform_scores) % uniform sampling over the columns
    uniform_scores = 1/n * ones(n, 1);
    [orthogonal_errors_uniform_scores, oblique_errors_uniform_scores] = compute_averages(A, ...
        uniform_scores, num_avg, ranks);
    fig_legend_string = [fig_legend_string, "$\sim 1/n$"];
end

if (show_column_scores) % sample proportionally to the columns norm of A
    column_scores = sum(A.^2, 2);
    [orthogonal_errors_columns_scores, oblique_errors_columns_scores] = compute_averages(A, ...
        column_scores / sum(column_scores), num_avg, ranks);
    fig_legend_string = [fig_legend_string, "$\sim \vert \vert a_j \vert \vert_2^2$"];
end

%% plot results with orthogonal projection error
fig = figure();
x = (1:ranks);
semilogy(x, gold_standards(2:ranks+1), 'LineWidth', 2.0);
hold on
if (show_leverage_scores)
    semilogy(x, orthogonal_errors_leverage_scores, 'LineWidth', 2.5);
    hold on
end
if (show_ridge_scores)
    semilogy(x, orthogonal_errors_ridge_scores, 'LineWidth', 2.5);
    hold on
end
if (show_estimated_ridge_scores)
    semilogy(x, orthogonal_errors_estimated_ridge_scores, 'LineWidth', 2.5);
    hold on
end
if (show_uniform_scores)
    semilogy(x, orthogonal_errors_uniform_scores, 'LineWidth', 2.5);
    hold on
end
if (show_column_scores)
    semilogy(x, orthogonal_errors_columns_scores, 'LineWidth', 2.5);
    hold on
end
xlabel("k", 'FontSize', 12);
ylabel("$\vert \vert A - Q Q^T A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Low-rank approximation by column sampling", 'FontSize', 12);
legend(fig_legend_string, 'interpreter', 'latex');
legend('Location', 'southwest', 'FontSize', 15, 'NumColumns', 3);
saveas(fig, "../figures/orthogonal_plot", "png");
saveas(fig, "../figures/orthogonal_plot", "epsc");

%% plot results with oblique projection error
fig2 = figure();
x = (1:ranks);
semilogy(x, gold_standards(2:ranks+1), 'LineWidth', 2.0);
hold on
if (show_leverage_scores)
    semilogy(x, oblique_errors_leverage_scores, 'LineWidth', 2.5);
    hold on
end
if (show_ridge_scores)
    semilogy(x, oblique_errors_ridge_scores, 'LineWidth', 2.5);
    hold on
end
if (show_estimated_ridge_scores)
    semilogy(x, oblique_errors_estimated_ridge_scores, 'LineWidth', 2.5);
    hold on
end
if (show_uniform_scores)
    semilogy(x, oblique_errors_uniform_scores, 'LineWidth', 2.5);
    hold on
end
if (show_column_scores)
    semilogy(x, oblique_errors_columns_scores, 'LineWidth', 2.5);
    hold on
end
xlabel("k", 'FontSize', 12);
ylabel("$\vert \vert A - C C^\dagger A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Low-rank approximation by column sampling", 'FontSize', 12);
legend(fig_legend_string, 'interpreter', 'latex');
legend('Location', 'southwest', 'FontSize', 15, 'NumColumns', 3);
saveas(fig2, "../figures/oblique_plot", "png");
saveas(fig2, "../figures/oblique_plot", "epsc");