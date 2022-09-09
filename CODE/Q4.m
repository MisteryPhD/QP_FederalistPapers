close all;
clear all;
clc;
 
%% Preparations
    % Load the data
[train,tune,test,dataDim] = getFederalistData;

    % Load the FederallistData again to get
    % the "wordlist"
load federalData;

    % Parse the data
y = [train(:,1); tune(:,1)];
y(y==2)=-1;
x = [train(:,2:end); tune(:,2:end)]';
 
    %
%% Test
    % Prepare M and H matrices    
M = x(:,y==-1); % M is the set of objects of 1st class (Madison)
H = x(:,y==1);  % H is the set of objects of 2nd class (Hamilton)
 
    % Specify mu value
mu = 0.1;
    % Specify i and j - features (two features from all 70 features set)
i = 1;
j = 60;
    % Obtain optimal w,b SVM parameters
[z,b,w,p1,p2] = run_NOVEL(M([i j],:),H([i j],:),mu);


 
%% Plot resulting 2D "SVM mapping"

    % At first estimate new x,y limits
    % (to make figure a bit clearer)
x_min = min([M(i,:) H(i,:) test(:,i)']);
x_max = max([M(i,:) H(i,:) test(:,i)']);
x_width = x_max - x_min;
new_x_min = x_min - x_width*0.2;
new_x_max = x_max + x_width*0.2;

y_min = min([M(j,:) H(j,:) test(:,j)']);
y_max = max([M(j,:) H(j,:) test(:,j)']);
y_width = y_max - y_min;
new_y_min = y_min - y_width*0.2;
new_y_max = y_max + y_width*0.2;

figure;
hold on;
    % Plot all objects
    % ('o' - Hamilton, '+' - Medison and '*' - disputed)
scatter(M(i,:),M(j,:),'+');
scatter(H(i,:),H(j,:),'o');
scatter(test(:,i),test(:,j),'*');
    % Plot '2D-hyperplane - line'
line([new_x_min new_x_max],[-(b+new_x_min*(w(1)/w(2))) -(b+new_x_max*(w(1)/w(2)))],'Color','r','LineWidth',2);    
    % Set new limits for x,y
xlim([new_x_min new_x_max]);
ylim([new_y_min new_y_max]);
    % Set appropriate labels and title
xlabel(wordlist(i));
ylabel(wordlist(j));
title('SVM, 2D mapping');
    % Add legend
legend({'Madison','Hamilton','disputed','SVM hyperplane'});