function result = ERCRD(data, n_samples, n_ensembles, lambda)
%   Collaborative-Representation-Based Detector (Ensemble & Robust)
%
% Usage
%   [result] = ERCRD(Data, ratio, lambda)
% Inputs
%   Data - 3D data matrix (num_row x num_col x num_dim)
%   n_samples - Randomly choose n_samples points to run Robust_CRD
%   n_ensembles - Ensemble learning bagging numbers
%   lambda - regularization parameter
% Outputs
%   result - Detector output scores (num_row x num_col)

[w, h, p] = size(data);
n = w * h;
s = n_samples;

scores = zeros(1, n);
X = reshape(data, n, p)';


%%
for ensemble = 1: n_ensembles
    randidx = randperm(n);
    X_s = X(:, randidx(1:s));
    W = solve_L21(X, X_s, lambda);
    resi = X - X_s * W;
    
    % Euclidean distance
    for i = 1: n
        scores(i) = scores(i) + resi(:, i)'*resi(:, i);
    end
end
result = scores;
end