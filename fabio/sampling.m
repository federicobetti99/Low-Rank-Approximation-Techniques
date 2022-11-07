%Sampling function

function k = sampling(p)
P = cumsum(p);
k = nnz(rand>P)+1;
end

