figure;

subplot(2,1,1);
hold on
plot(error_train,'b');
plot(error_tune,'r');
legend({'Train','Tune'});
xlabel('k');
ylabel('Error, %');

subplot(2,1,2);
plot(Goal_values);
xlabel('k');
ylabel('Goal function');