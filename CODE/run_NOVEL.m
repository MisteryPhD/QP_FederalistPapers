function [z,b,w,p1,p2] = run_NOVEL(M,H,mu)
    %% Preparations
        % Parse the data
            % training set 
    x = [M(:,11:end) H(:,11:end)];
    y = [-ones(size(M(:,11:end),2),1); ones(size(H(:,11:end),2),1)];
            % tuning set (for tuning set - 10 points from M and 10 from H)
    x_tune = [M(:,1:10) H(:,1:10)];
    y_tune = [-ones(size(M(:,1:10),2),1); ones(size(H(:,1:10),2),1)];
 
        % Initial assumption
    w = zeros(size(x,1),1);
    b = 0;
 
        % Initialize solution/solver
    t = 0;
    
        % Define the objective function
    Goal = @(y,x,mu,w,b)(1/length(y)) * sum( max([zeros(1,length(y));1-(y').*(w'*x+b)]) ) + (mu/2)*(w')*w;
    
%% Solving
Goal_before = 0;
for iter_count = 1:10000    
    % Each step, the algorithm will run stochastic gradient descent
    % stepping all objects in random order. Here the order given by I.
    I = randperm(length(y));
    
    for j=1:length(y)
        t = t+1;
        % Learning rate decreasing with each stochastic step
        % (with initial value of 1/mu).
        etta = 1.0/(mu*t);
        
        % Error flag defines which sub-gradient would be used
        error_flag = (1-(y(I(j)))*(w'*x(:,I(j))+b))>0;
 
        % Gradient step for all "w" elements
        for i = 1:size(w,1)
            % On respect to error_flag one or another sub-gradient would be 
            % chosen.
            if(error_flag)
                w(i) = w(i) - etta * (-y(I(j))*x(i,I(j)) + mu*w(i));
            else
                w(i) = w(i) - etta * ( mu*w(i));
            end
        end
        
        % Gradient step for "b"
        if(error_flag)
            b = b - etta*(-y(I(j)));
        end
    end
    
    Goal_then = Goal(y,x,mu,w,b);
    if(abs((Goal_before - Goal_then))/Goal_before <10^(-6))
        break;
    end
    Goal_before = Goal_then;
 
end
 %% Finalization   
    % Estimate requested z, p1, p2.
        % At first, estimate indentations
    M_full = (y').*((w')*x+b);
        % Then errors
    E = max([(1 - M_full); zeros(size(y'))]);
        % Finally, z - objective function
    z = (1/length(y))*sum(E) + (mu/2)*(w'*w);
        % misclassified points - points with negative indentation
        % (here comparison with "machine zero" is done instead of
        % real zero)
    p1 = sum(M_full < 1e-10);        
        % estimate misclassified points for tuning set - 
        % points with negative indentation
        % (here comparison with "machine zero" is done instead of
        % real zero)
    M_full = (y_tune').*((w')*x_tune+b);
    p2 =  sum(M_full < 1e-10);
end
