x = 0:3000;
M1 = csvread('Warm_sheet-EC_training.csv');
data0=M1(1:40,1);
pd = fitdist(data0,'wbl');
y = pdf(pd,x);
plot(x,y)
hold on
Test = csvread('Warm_sheet-EC_Testing.csv');
testdata1 = Test(1:20,1);
pd = fitdist(testdata1,'wbl');
y = pdf(pd,x);
plot(x,y)
legend('Warm-Train','Warm-Test')
nbins = 15;
figure()
h = histogram(data0,nbins)
count = 0;
for i=1:length(testdata1)
    if((testdata1(i)<1300 && testdata1(i)>800))
      count = count +1;
    end
end
count/20