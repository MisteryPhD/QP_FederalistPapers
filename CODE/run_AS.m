function [z,b,w,p1,p2] = run_AS(M,H,mu)
    %% Preparations
        % Parse the data
            % training set 
    x = [M(:,11:end) H(:,11:end)];
    y = [-ones(size(M(:,11:end),2),1); ones(size(H(:,11:end),2),1)];
            % tuning set (for tuning set - 10 points from M and 10 from H)
    x_tune = [M(:,1:10) H(:,1:10)];
    y_tune = [-ones(size(M(:,1:10),2),1); ones(size(H(:,1:10),2),1)];
    
        % Prepare Q matrix
    Q = (y*y').*(x'*x);
 
        % Initial assumption
            % Support vectors
    I_s = zeros(size(y,1),1);
            % Periphery vectors
    I_o = ones(size(y,1),1);
            % Intruder vector
    I_c = zeros(size(y,1),1);
 
            % Find initial support vectors
                % 1)Take some random object and store it in i1
    i1 = randi([1 length(I_s)]);
    i1_previous = 0;
                % (not more then 100 steps for this finding procedure,
                % usually it is converged by few iterations)
    for l=1:100
                % 2)Estimate the distance between all other-class-objects to
                % to the i1 object
        y_n_ind = find(y~=y(i1));
        r = sum((x(:,y_n_ind)-x(:,i1)).^2);
                % 3)Choose as optimal pair for i1 such i2 object that
                % is located on minimal distance from i1.
        i2 = y_n_ind(find(r==min(r),1));
                % 4)If convergence reached - break the process...
        if(i1_previous == i2)
            break;
        end
                % ... if convergence not reached yet - continue the process
                % for i2 object.
        i1_previous = i1;
        i1 = i2;    
    end
 
            % Use found i1 and i2 pair of objects from different classes
            % as initialization for S set.
    I_s([i1 i2]) = 1;
    I_o([i1 i2]) = 0;
 
        % Define C
    C = 1 / (mu * size(y,1));  
 
        % Variable for iterations count
    iter_count = 0;
 
    %% Solving
    flag2 = 1;
 
    while flag2
        % In case of "endless" cycling - break it
        if(iter_count>=10000)
            break;
        end
 
        flag1 = 1;
        %fprintf(1,'=========\n');
        while flag1
                % Construct S/C matrices
            e_s = ones(sum(I_s),1);
            e_c = ones(sum(I_c),1);
            Q_cs = Q(I_c==1, I_s==1);
            Q_ss = Q(I_s==1, I_s==1);
 
                % lamda optimization
            lamda_s = 2 * (e_s' - C*(e_c')*Q_cs) * (Q_ss^(-1));
 
            flag1 = 0;
 
            i_s = find(I_s==1);
                % Decompose vectors
            if(sum(I_s)>2)
                    % From support vector to periphery
                i_s_0 = i_s(lamda_s <= 0);
                if(size(i_s_0,1)>0)
                    % Chose object for decomposition (between S and O sets)
                    % randomly but with modulated probability p_s_0.
                    % The probability function p_s_0 is chosen in such way
                    % that objects that strongly violates "lamda_s<0"
                    % condition will be chosen with higher probability
                    % (it has higher priority).
                    lamda_s_0 = lamda_s(lamda_s<=0);
                    p_s_0 = 0.5*(ones(size(lamda_s_0))/(length(lamda_s_0))) +...
                            0.5 * ((0-lamda_s_0)/sum(0-lamda_s_0));          
                    i_s_0_chosen = i_s_0(1+sum(rand >= cumsum(p_s_0)));
 
                    I_s(i_s_0_chosen) = 0;
                    I_o(i_s_0_chosen) = 1;
        %            fprintf(1,'S->O: %d \n',i_s_0_chosen);
                    flag1 = 1;
                    iter_count = iter_count + 1;
                        % If some decompositions happened
                        % continue the cycle
                    continue;
                end
 
                    % From support vector to intruder
                i_s_C = i_s(lamda_s >= C);
 
                if(size(i_s_C,1)>0)
                    lamda_s_C = lamda_s(lamda_s>=C);
                    p_s_C = 0.5*(ones(size(lamda_s_C))/(length(lamda_s_C))) +...
                            0.5*((lamda_s_C-C) / sum(lamda_s_C-C));          
 
                    i_s_C_chosen = i_s_C(1+sum(rand >= cumsum(p_s_C)));
                    I_s(i_s_C_chosen) = 0;
                    I_c(i_s_C_chosen) = 1;
         %           fprintf(1,'S->C: %d \n',i_s_C_chosen);
                    flag1 = 1;
                    iter_count = iter_count + 1;
                    continue;
                end
            end
 
        end
     %   fprintf(1,'++++++++\n');
            % Estimate SVM coefficients
        w = x(:,I_s==1) * ((lamda_s').*y(I_s==1));
        b = median(y(I_s==1)' - (w')*x(:,I_s==1));
 
        flag2 = 0;
 
        i_o = find(I_o==1);
        i_c = find(I_c==1);
            % Estimate indent
        M_o = (y(i_o)') .* ((w')*x(:,i_o) + b);
        M_c = (y(i_c)') .* ((w')*x(:,i_c) + b);
 
            % Decompose vectors
 
            % From periphery to support vector
        i_o_less = i_o(M_o <= 1);
 
        if(size(i_o_less,1)>0)
            M_o_less = M_o(M_o<=1);
            p_o_less = 0.5*(ones(size(M_o_less))/(length(M_o_less))) +...
                    0.5 * ((1- M_o_less)/sum(1- M_o_less));                           
            i_o_less_chosen = i_o_less(1+sum(rand >= cumsum(p_o_less)));
 
            I_o(i_o_less_chosen) = 0;
            I_s(i_o_less_chosen) = 1;
        %    fprintf(1,'O->S: %d \n',i_o_less_chosen);
            flag2 = 1;
            iter_count = iter_count + 1;
                % If some decompositions happened
                % continue the cycle
            continue;
        end
 
            % From intruder to support vector
        i_c_more = i_c(M_c >= 1);
 
        if(size(i_c_more,1)>0)
            M_c_more = M_c(M_c>=1);
            p_c_more = 0.5*(ones(size(M_c_more))/(length(M_c_more))) +...
                       0.5 * ((M_c_more-1)/sum(M_c_more-1));                           
            i_c_more_chosen = i_c_more(1+sum(rand >= cumsum(p_c_more)));
 
            I_c(i_c_more_chosen) = 0;
            I_s(i_c_more_chosen) = 1;
      %      fprintf(1,'C->S: %d \n',i_c_more_chosen);
            flag2 = 1;
            iter_count = iter_count + 1;
            continue;
        end
 
 
    end
    
    % Estimate requested z, p1, p2.
        % At first, estimate indentations
    M_full = (y').*((w')*x+b);
        % Then errors
    E = max([(1 - M_full); zeros(size(y'))]);
        % Finally, z - objective function
    z = (1/length(y))*sum(E) + (mu/2)*(w'*w);
        % missclassified points - points with negative indentation
        % (here comparision with "machine zero" is done instead of
        % real zero)
    p1 = sum(M_full < 1e-10);
        % estimate misclassified points for tuning set - 
        % points with negative indentation
        % (here comparison with "machine zero" is done instead of
        % real zero)
    M_full = (y_tune').*((w')*x_tune+b);
    p2 =  sum(M_full < 1e-10);
end
