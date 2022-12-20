%% clean
clear
close all
clc

%% define Hilbert matrix and compute all scores for sketching
n = 100;
A = hilb(n);
[U, S, V] = svd(A);

fig_legend_string = ["$\propto l_{i, \lambda}(A)$", "$\propto 1/n$", ...
                     "$\propto \| a_j \|_2^2$"];

lambda = 1e-4; % regularization parameter
ridge_scores = diag(V * diag(diag(S).^2 ./ (diag(S).^2 + lambda^2)) * V'); % compute ridge scores
ridge_scores = ridge_scores / sum(ridge_scores);

column_scores = sum(A.^2, 2) / norm(A, "fro")^2;  % compute columns norm scores

uniform_scores = 1/n * ones(n, 1);  % compute uniform scores

%% plot cumulative sums
fig = figure();
plot(cumsum(ridge_scores), 'LineWidth', 2.5);
hold on
plot(cumsum(uniform_scores), 'LineWidth', 2.5);
hold on
plot(cumsum(column_scores), 'LineWidth', 2.5);
xlabel("Column index", 'FontSize', 12);
ylabel("Cumulative sum", 'FontSize', 12);
ax = gca;
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title("Cumulative sum of sketching distributions", 'FontSize', 12);
legend(fig_legend_string, 'interpreter', 'latex');
legend('Location', 'southeast', 'FontSize', 15, 'NumColumns', 1);
saveas(fig, "../figures/dist_scores", "epsc");