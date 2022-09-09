close all;
clear all;
clc;

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

    % Specify mu value
mu = 0.1;
    % Define the array to store the tuning missclassification
    % (there are 20 points in tuning set, defaul value for p2 would 20)
p2_q = 20*ones(size(M,1),size(M,1));
z_q = 10000*ones(size(M,1),size(M,1));
for i=1:size(M,1)
    for j=(i+1):size(M,1)
            % Test by run_AS
        [z_q(i,j),b,w,p1,p2_q(i,j)] = run_quadprog(M([i j],:),H([i j],:),mu);
    end
end


%% Output testing results
