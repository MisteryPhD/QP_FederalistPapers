close all;
clear all;
clc;

% Load the data
[train,tune,test,dataDim] = getFederalistData;

% Parse the data
N = size(train,1);
n = size(train,2)-1;
y = train(:,1);
y(y==2)=-1;
x = train(:,2:end)';

y_tune = tune(:,1);
y_tune(y_tune==2)=-1;
x_tune = tune(:,2:end)';

% Define guess
w = zeros(n,1);
b = 0;

mu = 1;

% Define the objective function
Goal = @(y,x,mu,w,b)(1/N) * sum( max([zeros(1,N);1-(y').*(w'*x+b)]) ) + (mu/2)*(w')*w;

% Initialize solution/solver
t = 0;

break_flag = 0;
for iter = 1:10000
    I = randperm(length(y));
    Goal_before = Goal(y,x,mu,w,b);
    for j=1:length(y)
        
        t = t+1;    
        etta = 1.0/(mu*t);
        
        Goal_values(t) = Goal(y,x,mu, w,b);
        error_flag = 1-(y(I(j)))*(w'*x(:,I(j))+b)>0;

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

        % Estimate tune errors
        error_train(t) = 100*(1-sum(y==(((w'*x+b)>=0)*2-1)')/size(y,1));
        error_tune(t) = 100*(1-sum(y_tune==(((w'*x_tune+b)>=0)*2-1)')/size(y_tune,1));
    

    end
    if(abs((Goal_before - Goal(y,x,mu,w,b)))/Goal_before <10^(-6))
            break;
    end
end

% Plot train\tune errors as function of gradient descent step
figure;

subplot(2,1,1);
hold on
plot(error_train,'b');
plot(error_tune,'r');
legend({'Train','Tune'});
xlabel('k');
ylabel('Error, %');

subplot(2,1,2);
semilogy(Goal_values);
xlabel('k');
ylabel('Goal function');