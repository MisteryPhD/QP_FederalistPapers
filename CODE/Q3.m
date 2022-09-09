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
    % Define the array to store the tuning misclassification
    % (there are 20 points in tuning set, default value for p2 would 20)
p2 = 20*ones(size(M,1),size(M,1));
z = 10000*ones(size(M,1),size(M,1));
for i=1:size(M,1)
    for j=(i+1):size(M,1)
            % Test by run_*
        [z(i,j),b,w,p1,p2(i,j)] = run_AS(M([i j],:),H([i j],:),mu);
    end
end
 
 
%% Output testing results
    % Estimate
j_min = ceil(find(p2_n==min(min(p2_n)),1)/70);
i_min = find(p2_n==min(min(p2_n)),1)-70*(j_min-1);
 
    % Draw tuning misclassifications as image
figure;
imagesc(p2_n);
hold on;
rectangle('Position',[(j_min-3) (i_min-3) 5 5], 'Curvature',[1 1],'EdgeColor','r','LineWidth',2);
xlabel('j');
ylabel('i');
title('Misclassified tuning points');
 
    % Output optimal features numbers and corresponding tuning
    % misclassifications
fprintf(1,'i= %d, j= %d, p2= %d\n',i_min, j_min, p2_n(i_min,j_min));    
