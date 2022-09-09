function [z,b,w,p1,p2] = run_quadprog(M,H,mu)
    % Parse the data
        % training set 
    x = [M(:,11:end) H(:,11:end)];
    y = [-ones(size(M(:,11:end),2),1); ones(size(H(:,11:end),2),1)];
            % tuning set (for tuning set - 10 points from M and 10 from H)
    x_tune = [M(:,1:10) H(:,1:10)];
    y_tune = [-ones(size(M(:,1:10),2),1); ones(size(H(:,1:10),2),1)];
    
            % Estimate C
    C = 1 / (mu * size(y,1));
 
        % Parse 'SVM' data into QP problem
    Q = x'*x;
    Y = diag(y);
    H = Y*Q*Y + 1e-6*eye(length(y));
    f = -ones(size(y));
    a = y';
    K = 0;
    Kl = zeros(size(y));
    Ku = C * ones(size(y));
 
        % Solve by quadprog
    options = optimset('Display','off');    
    lamda = quadprog(H,f,[],[],a,K,Kl,Ku,[],options);
 
    w = x * (lamda.*y);
 
    e = 1e-6;
    ind = find(lamda > e & lamda < C-e);
    b = mean(y(ind) - x(:,ind)'*w);
 
            % To estimate objective function, at first
            % estimate indents M
    M_full = (y').*((w')*x+b);
            % Then errors
    E = max([(1 - M_full); zeros(size(y'))]);
            % Finally, z - objective function
    z = (1/length(y))*sum(E) + (mu/2)*(w'*w);
            % misclassified points - points with negative margin
            % (here comparison with "machine zero" is done instead of
            % real zero)
    p1 = sum(M_full < 1e-10);
    
            % Make the same for tuning set
    M_full = (y_tune').*((w')*x_tune+b);
    p2 =  sum(M_full < 1e-10);

end
