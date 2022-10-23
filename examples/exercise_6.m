clear
close all
clc

%% first matrix
tic
n = 100;
A = hilb(n);
errors = zeros(rank(A), 1);
for k=1:rank(A)
   Tau = greedy_rank_approximation(A, k);
   errors(k) = norm(A-Tau);
end
toc

[~, S, ~] = svd(A);
figure()
loglog(1:rank(A), diag(S(2:rank(A)+1, 2:rank(A)+1)), 'LineWidth', 1);
hold on
loglog(1:rank(A), errors, 'LineWidth', 2);
legend("$\vert \vert A - T_r(A) \vert \vert_2$", "$\vert \vert A-A_r \vert \vert_2$", "FontSize", 15, ...
       "Location", "southwest", "interpreter", "latex", "NumColumns", 1);


%% second matrix
tic
n = 100;
A = zeros(n, n);
for i=1:n
    for j=1:n
        A(i,j) = exp(-abs(i-j)/1000);
    end
end

errors = zeros(rank(A), 1);
for k=1:rank(A)
   Tau = greedy_rank_approximation(A, k);
   errors(k) = norm(A-Tau, 2);
end
toc

[~, S, ~] = svd(A);
figure()
loglog(diag(S(2:rank(A), 2:rank(A))), 'LineWidth', 1);
hold on
loglog(errors(1:end-1), 'LineWidth', 2);
legend("$\vert \vert A - T_r(A) \vert \vert_2$", "$\vert \vert A-A_r \vert \vert_2$", "FontSize", 15, ...
       "Location", "northeast", "interpreter", "latex", "NumColumns", 1);
