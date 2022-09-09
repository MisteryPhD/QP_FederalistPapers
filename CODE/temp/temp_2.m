i1 = randi([1 length(I_s)]);
i1_previous = 0;
while 1
    y_n_ind = find(y~=y(i1));
    r = sum((x(:,y_n_ind)-x(:,i1)).^2);
    i2 = y_n_ind(find(r==max(r),1));
    fprintf(1,'%d %d\n',i1,i2);
    if(i1_previous == i2)
        break;
    end
    i1_previous = i1;
    i1 = i2;    
end
