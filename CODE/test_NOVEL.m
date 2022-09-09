function [z, b, p1, p2] = test_NOVEL(mu)
    %% Preparations
        % Load the data
    [train,tune,test,dataDim] = getFederalistData;

        % Parse the data
    y = [train(:,1); tune(:,1)];
    y(y==2)=-1;
    x = [train(:,2:end); tune(:,2:end)]';

    %% Test
        % Prepare M and H matrices    
    M = x(:,y==-1); % M is the set of objects of 1 class (Madison)
    H = x(:,y==1);  % H is the set of objects of 2 class (Hamilton)

        % test by run_NOVEL
    tic;   
    [z,b,w,p1,p2] = run_NOVEL(M,H,mu);
    time = toc;
    
        % Estimate train/tune errors
    error_train = 100*(p1 / 86);
    error_tune = 100*(p2 / 20);
    
        % Estimate w2
    w2 = w'*w;
    
        % Classify disputed papers
    c = (((w'*test'+b)>=0)*2-1)';

    %% Output testing results
    fprintf(1,'Train error: %2.2f %%\n',error_train);
    fprintf(1,'Tune error: %2.2f %%\n',error_tune);

    fprintf(1,'z= %f, b= %2.2f, ||w||2= %f, p1= %d, p2= %d\n',z,b,w2,p1,p2);
    
    fprintf(1,'Elapsed time: %f s\n',time);
    
    fprintf(1,'Disputed papers authorship prediction:\n');
    fprintf(1,'%d\n',c);
end