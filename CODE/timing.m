clear all;
close all;
clc;

%% Preparations
% Load the data
[train,tune,test,dataDim] = getFederalistData;

        % Parse the data
    y = [train(:,1); tune(:,1)];
    y(y==2)=-1;
    x = [train(:,2:end); tune(:,2:end)]';
    
    M = x(:,y==-1); % M is the set of objects of 1 class (Madison)
    H = x(:,y==1);  % H is the set of objects of 2 class (Hamilton) 
    
    x = [M(:,11:end) H(:,11:end)];
    y = [-ones(size(M(:,11:end),2),1); ones(size(H(:,11:end),2),1)];
            % tuning set (for tuning set - 10 points from M and 10 from H)
    x_tune = [M(:,1:10) H(:,1:10)];
    y_tune = [-ones(size(M(:,1:10),2),1); ones(size(H(:,1:10),2),1)];

%% Mine the statistics data
k = 10:1000;
mu = 1;
time_AS = zeros(size(k));
error_train_AS = zeros(size(k));
error_tune_AS = zeros(size(k));
z_AS = zeros(size(k));

time_NOVEL = zeros(size(k));
error_train_NOVEL = zeros(size(k));
error_tune_NOVEL = zeros(size(k));
z_NOVEL = zeros(size(k));

for i=1:length(k)
    for j=1:10
        % Test AS with given k-steps constraint
        [b,w,time_k] = timing_AS(x,y,mu,k(i));
        time_AS(i) = time_AS(i)+ time_k;
        error_train_AS(i) = error_train_AS(i) + 100*(1-sum(y==(((w'*x+b)>=0)*2-1)')/size(y,1));
        error_tune_AS(i) = error_tune_AS(i) + 100*(1-sum(y_tune==(((w'*x_tune+b)>=0)*2-1)')/size(y_tune,1));
        M = (y').*((w')*x+b);
        E = max([(1 - M); zeros(size(M))]);
        z_AS(i) = z_AS(i) + (1/length(y))*sum(E) + (mu/2)*(w'*w);
    end
    time_AS(i) = time_AS(i)/10;
    error_train_AS(i) = error_train_AS(i)/10;
    error_tune_AS(i) = error_tune_AS(i)/10;
    z_AS(i) = z_AS(i)/10;

    for j=1:10
        % Test AS with given k-steps constraint
        [b,w,time_k] = timing_NOVEL(x,y,mu,k(i));
        time_NOVEL(i) = time_NOVEL(i)+ time_k;
        error_train_NOVEL(i) = error_train_NOVEL(i) + 100*(1-sum(y==(((w'*x+b)>=0)*2-1)')/size(y,1));
        error_tune_NOVEL(i) = error_tune_NOVEL(i) + 100*(1-sum(y_tune==(((w'*x_tune+b)>=0)*2-1)')/size(y_tune,1));
        M = (y').*((w')*x+b);
        E = max([(1 - M); zeros(size(M))]);
        z_NOVEL(i) = z_NOVEL(i) + (1/length(y))*sum(E) + (mu/2)*(w'*w);
    end
    time_NOVEL(i) = time_NOVEL(i)/10;
    error_train_NOVEL(i) = error_train_NOVEL(i)/10;
    error_tune_NOVEL(i) = error_tune_NOVEL(i)/10;
    z_NOVEL(i) = z_NOVEL(i)/10;
end

%% Present mined results (plot figures)

    % Plot AS figures
figure;

subplot(3,1,1);
hold on;
title('AS testing');
plot(k,error_train_AS,'b');
plot(k,error_tune_AS,'r');
ylabel('Error, %');
legend({'Train','Tune'});


subplot(3,1,2);
plot(k,time_AS);
ylabel('time, s');

subplot(3,1,3);
semilogy(k,z_AS);
xlabel('k');
ylabel('Objective function');


    % Plot NOVEL figures
figure;

subplot(3,1,1);
hold on;
title('NOVEL testing');
plot(k,error_train_NOVEL,'b');
plot(k,error_tune_NOVEL,'r');
ylabel('Error, %');
legend({'Train','Tune'});


subplot(3,1,2);
plot(k,time_NOVEL);
ylabel('time, s');

subplot(3,1,3);
semilogy(k,z_NOVEL);
xlabel('k');
ylabel('Objective function');