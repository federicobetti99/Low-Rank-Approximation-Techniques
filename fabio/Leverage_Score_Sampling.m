
% LOW-RANK APPROXIMATION TECHNIQUE PROJECT

clc

%Problem data
lambda = 1e-4;
N = 100;
A = hilb(N); %initialize Hilbert matrix
disp(A); %add more disp throughout the file

[U, S, V] = svd(A); %do a not full order svd?
Sigma = diag(S);
disp( U)
disp(Sigma)
disp( V);

ranks = 50;

%Ridge Leverage Scores

scores = zeros(N,1);
matrix_lev = diag( (diag(S).^2) ./ (diag(S).^2 + lambda^2));
for j=1:N
    scores(j) = V(j,:)* matrix_lev * V(j,:)'; %by point d of Exercise 1
end

ridge_sum = sum(scores);

disp(scores)
disp(ridge_sum)

ridge_scores = scores ./ ridge_sum;

% Uniform scores

uniform_scores = ones(N,1)./N;

% Column scores

columns_scores = zeros(N, 1);
for k=1:N
   column_scores(k) = (norm(A(:, k)) / norm(A, "fro"))^2;
end


%%%Multiple-run average

%Computation errors Ridge Leverage Score sampling

err_ridges_pseudo= zeros(20, ranks);
err_ridges = zeros(20,ranks);

for m=1:20
    for k=1:ranks
        C = zeros(N,k);
        for l=1:k
            p = sampling(ridge_scores);
            C(:, l) = A(:, p) / sqrt(k * ridge_scores(p));  %unbiased column
        end
        err_ridges_pseudo(m,k) = norm(A-C*pinv(C)*A, 2);  % ||A-CC^(daga)||_2, pinv(C1)= pseudoinverse of C1
        [Uk, ~, ~] = svd(C, 0);  % thin svd
        Qk = Uk(:, 1:k);  % left principal subspace up to K
        err_ridges(m, k) = norm(A-Qk*Qk'*A, 2);  %  ||A-QkQk.T A||_2 
    end
end

%Computation errors Uniform Sampling

err_uniform_pseudo = zeros(20, ranks);  
err_uniform = zeros(20,ranks);

for m=1:20
    for k=1:ranks
       C = zeros(N, k);
       for l=1:k
           p = sampling(uniform_scores);  
           C(:, l) = A(:, p) / sqrt(k * uniform_scores(p));  % unbiased column
       end
       err_uniform_pseudo(m, k) = norm(A-C*pinv(C)*A, 2);  % ||A-CC^(daga)||_2 
       [Uk, ~, ~] = svd(C, 0);  % thin svd
       Qk = Uk(:, 1:k);  % left principal subspace up to K
       err_uniform(m, k) = norm(A-Qk*Qk'*A, 2);  %  ||A-QkQk.T A||_2 
    end
end


% Computation Errors according to Column sampling

err_columns_pseudo = zeros(20, ranks);
err_columns = zeros(20, ranks);

for m=1:20
    for k=1:ranks
       C = zeros(N, k);
       for l=1:k
           S = sampling(column_scores);  
           C(:, l) = A(:, S) / sqrt(k * column_scores(S));  % unbiased column
       end
       err_columns_pseudo(m, k) = norm(A-C*pinv(C)*A, 2);  % ||A-CC^(daga)||_2
       [Uk, ~, ~] = svd(C, 0);  % thin svd
       Qk = Uk(:, 1:k);  % left principal subspace up to K
       err_columns(m, k) = norm(A-Qk*Qk'*A, 2);  %  ||A-QkQk.T A||_2 
    end
end


%%%%%%%%% Plotting errors outcomes

%||A-CC^(daga)||_2, 
average_err_ridges_pseudo = mean(err_ridges_pseudo);
average_err_uniform_pseudo = mean(err_uniform_pseudo);
average_err_columns_pseudo = mean(err_columns_pseudo);


fig1 = figure();
x = (1:ranks);
semilogy(x, average_err_ridges_pseudo,'LineWidth', 1.0);
hold on
semilogy(x, average_err_uniform_pseudo,'LineWidth', 1.0);
hold on
semilogy(x, average_err_columns_pseudo, 'LineWidth', 1.0);
hold on
semilogy(x, Sigma(2:ranks+1),'LineWidth', 1.0);


xlabel("k", 'FontSize', 11);
ylabel("$\vert \vert A - C C^{\dagger} A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 11);
ax = gca;

title("Low-rank approximation - different sampling", 'FontSize', 12);

legend("Ridge leverage scores", "Uniform sampling", "Column-norm sampling", "$\sigma_{k+1}(A)$", 'interpreter', 'latex');
legend('Location', 'best', 'FontSize', 12, 'NumColumns', 1);
saveas(fig1, 'plot', 'epsc');
%saveas(fig1, 'plot.jpg');


%||A-QkQk.T||_2

average_err_ridges = mean(err_ridges);
average_err_uniform = mean(err_uniform);
average_err_columns = mean(err_columns);


fig2 = figure();
x = (1:ranks);
semilogy(x, average_err_ridges,'LineWidth', 1.0);
hold on
semilogy(x, average_err_uniform,'LineWidth', 1.0);
hold on
semilogy(x, average_err_columns, 'LineWidth', 1.0);
hold on
semilogy(x, Sigma(2:ranks+1),'LineWidth', 1.0);


xlabel("k", 'FontSize', 11);
ylabel("$\vert \vert A - QkQk^{T}A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 11);
ax = gca;

title("Low-rank approximation - different sampling", 'FontSize', 12);

legend("Ridge leverage scores", "Uniform sampling", "Column-norm sampling", "$\sigma_{k+1}(A)$", 'interpreter', 'latex');
legend('Location', 'best', 'FontSize', 12, 'NumColumns', 1);
%saveas(fig2, 'plot2', 'jpg');
saveas(fig2, 'plot2', 'epsc');