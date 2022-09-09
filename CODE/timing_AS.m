function [b,w,time] = timing_AS(x,y,mu,k)
 
    tic
    %% Preparations
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
    i1 = randi([1 length(I_s)]);
    i1_previous = 0;
    while 1
        y_n_ind = find(y~=y(i1));
        r = sum((x(:,y_n_ind)-x(:,i1)).^2);
        i2 = y_n_ind(find(r==min(r),1));
        if(i1_previous == i2)
            break;
        end
        i1_previous = i1;
        i1 = i2;    
    end
 
    I_s([i1 i2]) = 1;
    I_o([i1 i2]) = 0;
 
        % Define C    
    C = 1 / (mu * size(y,1));  
 
        % Variable for iterations count
    iter_count = 0;
 
    %% Solving
    flag2 = 1;
 
    while flag2
        flag1 = 1;
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
                    lamda_s_0 = lamda_s(lamda_s<=0);
                    p_s_0 = 0.5*(ones(size(lamda_s_0))/(length(lamda_s_0))) +...
                            0.5 * ((0-lamda_s_0)/sum(0-lamda_s_0));          
                    i_s_0_chosen = i_s_0(1+sum(rand >= cumsum(p_s_0)));
 
                    I_s(i_s_0_chosen) = 0;
                    I_o(i_s_0_chosen) = 1;
 
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
 
                    flag1 = 1;
                    iter_count = iter_count + 1;
                    continue;
                end
            end
 
        end
            % Estimate SVM coefficients
        w = x(:,I_s==1) * ((lamda_s').*y(I_s==1));
        b = median(y(I_s==1)' - (w')*x(:,I_s==1));
 
            % Break the optimization on "k" step
        if(iter_count>=k)
            break;
        end
        
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
            
            flag2 = 1;
            iter_count = iter_count + 1;
            continue;
        end
 
 
    end
    
 %% Finalization  
     % Estimate execution time
    time = toc;
end