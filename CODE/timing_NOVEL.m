function [b,w,time] = timing_NOVEL(x,y,mu,k)

    tic

 %% Preparations

        % Define the objective function
    Goal = @(y,x,mu,w,b)(1/length(y)) * sum( max([zeros(1,length(y));1-(y').*(w'*x+b)]) ) + (mu/2)*(w')*w;
 
        % Initial assumption
    w = zeros(size(x,1),1);
    b = 0;

        % Initialize solution/solver
    t = 0;
    
%% Solving
Goal_before = 0;
for iter_count = 1:10000
    I = randperm(length(y));
    
    for j=1:length(y)
        t = t+1;    
        etta = 1.0/(mu*t);
        
        error_flag = (1-(y(I(j)))*(w'*x(:,I(j))+b))>0;

        % Gradient step for all "w" elements
        for i = 1:size(w,1)
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
    
    if(iter_count>=k)
        break;
    end
end

    
 %% Finalization   
     % Estimate execution time
    time = toc;
end