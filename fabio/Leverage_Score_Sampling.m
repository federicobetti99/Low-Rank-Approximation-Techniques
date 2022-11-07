
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

err_ridge = zeros(20, ranks);

for m=1:20
    for k=1:ranks
        C = zeros(N,k);
        for l=1:k
            p = sampling(ridge_scores);
            C(:, l) = A(:, p) / sqrt(k * ridge_scores(p));  %unbiased column
        end
        err_ridges(m,k) = norm(A-C*pinv(C)*A, 2);  % ||A-CC^(daga)||_2, pinv(C1)= pseudoinverse of C1
    end
end

%Computation errors Uniform Sampling

err_uniform = zeros(20, ranks);  

for m=1:20
    for k=1:ranks
       C = zeros(N, k);
       for l=1:k
           p = sampling(uniform_scores);  
           C(:, l) = A(:, p) / sqrt(k * uniform_scores(p));  % unbiased column
       end
       err_uniform(m, k) = norm(A-C*pinv(C)*A, 2);  % ||A-CC^(daga)||_2 
    end
end


% Computation Errors according to Column sampling

err_columns = zeros(20, ranks);

for m=1:20
    for k=1:ranks
       C = zeros(N, k);
       for l=1:k
           S = sampling(column_scores);  
           C(:, l) = A(:, S) / sqrt(k * column_scores(S));  % unbiased column
       end
       err_columns(m, k) = norm(A-C*pinv(C)*A, 2);  % ||A-CC^(daga)||_2
    end
end


% Plotting errors outcomes


average_err_ridges = mean(err_ridges);
average_err_uniform = mean(err_uniform);
average_err_columns = mean(err_columns);


fig = figure();
x = (1:ranks);
semilogy(x, average_err_ridges,'LineWidth', 1.0);
hold on
semilogy(x, average_err_uniform,'LineWidth', 1.0);
hold on
semilogy(x, average_err_columns, 'LineWidth', 1.0);
hold on
semilogy(x, Sigma(2:ranks+1),'LineWidth', 1.0);


xlabel("k", 'FontSize', 11);
ylabel("$\vert \vert A - C C^{\dagger} A \vert \vert_2$", 'interpreter', 'latex', 'FontSize', 11);
ax = gca;

title("Low-rank approximation - different sampling", 'FontSize', 12);

legend("Ridge leverage scores", "Uniform sampling", "Column-norm sampling", "$\sigma_{k+1}(A)$", 'interpreter', 'latex');
legend('Location', 'best', 'FontSize', 12, 'NumColumns', 1);
saveas(fig, "plot", "epsc");
saveas(fig, "plot", "jpg");