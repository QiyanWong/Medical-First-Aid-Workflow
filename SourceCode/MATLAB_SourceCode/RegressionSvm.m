M1 = xlsread('data.xls','Sheet1','C2:C3919');
data0=M1(1:5,3);
X=M1(1:5,1);
Y=data0;
Mdl = fitrsvm(X,Y);

sum = 0;
iter = 1;
count = 0;

for i=1:length(M1)
    count = count+1;
    data0 = M1(iter:iter+5,3); 
    X = M1(iter:iter+5,1);
    Y =data0;
    Mdl = fitrsvm(X,Y);
    YHat = predict(Mdl,M1(iter+5,1));
     result = abs(YHat - M1(iter+5,3));
     prediction(count) = YHat;
     actual(count) = M1(iter+5,3);
     error(count) = result;
    if(iter > 3900) break;
    end
    iter = iter +7;
    sum = sum + result;
end
sum/count
S = std(error)
figure()
x1 = 0:500;
plot(prediction)
hold on
plot(actual)
legend('Predicted', 'Actual');
hold off
