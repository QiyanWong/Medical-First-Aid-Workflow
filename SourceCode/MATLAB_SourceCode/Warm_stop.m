x = 0:200;
M1 = csvread('Warm_sheet-EC_training.csv');
data0=M1(1:40,3);
pd = fitdist(data0,'wbl');
y = pdf(pd,x);
plot(x,y)
hold on
Test = csvread('Warm_sheet-EC_Testing.csv');
testdata1 = Test(1:20,3);
pd = fitdist(testdata1,'wbl');
y = pdf(pd,x);
plot(x,y)
legend('Warm-Train','Warm-Test')

count = 0;
for i=1:length(testdata1)
    if((testdata1(i)<100 && testdata1(i)>5))
      count = count +1;                                         
    end
end

count/20