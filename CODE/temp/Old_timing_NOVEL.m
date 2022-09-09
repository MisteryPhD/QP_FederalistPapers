function [b,w,time] = timing_NOVEL(x,y,mu,k)

    tic

 %% Preparations

        % Initial assumption
    w = ones(size(x,1),1);
    b = 0;

        % Initialize solution/solver
    d_current = ones(size(w,1)+1,1);
    g_current = ones(size(w,1)+1,1);
    lambda = 0.01;
    
%% Solving
    for iter_count = 1:1000
        error_flag = 1-(y').*(w'*x+b)>0;

        % Gradient step for all "w" elements
        for i = 1:size(w,1)
            g_current(i) = (1/length(y)) * sum((-y(error_flag)'.*x(i,error_flag))) + mu*w(i);
            d_current(i) = -lambda * g_current(i);
        end
        % Gradient step for "b"
        g_current(end) = (1/length(y)) * sum((-y(error_flag)'));
        d_current(end) = -lambda * g_current(end);

        w = w + d_current(1:(end-1));
        b = b + d_current(end);
        
        if(iter_count>=k)
            break;
        end
    end
    
 %% Finalization   
     % Estimate execution time
    time = toc;
end