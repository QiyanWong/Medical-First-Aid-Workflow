x = 0:3000;
M1 = csvread('Warm_sheet-EC_training.csv');
data0=M1;
pd = fitdist(data0,'wbl');
y = pdf(pd,x);
plot(x,y)
hold on
M2 = csvread('ABP_train.csv');
data=M2(1:15,1);
censored=M2(1:13,7);
Frequency = wblfit(censored);
pd = fitdist(data,'wbl');
y = pdf(pd,x);
plot(x,y)

M3 = csvread('T-spine_training.csv');
data1= M3;
pd = fitdist(data1,'wbl');
y = pdf(pd,x);
plot(x,y)
xlabel('Failure time');
ylabel('Hazard rate');
legend('Warm_sheet-EC_training','ABP_train','T-spine_training')
figure()
data=M2(16:30,7);
pd = fitdist(data,'wbl');
y = pdf(pd,x);
plot(x,y)
hold on
data=M2(30:45,7);
pd = fitdist(data,'wbl');
y = pdf(pd,x);
plot(x,y)
data=M2(46:59,7);
pd = fitdist(data,'wbl');
y = pdf(pd,x);
plot(x,y)
data=M2(60:74,7);
pd = fitdist(data,'wbl');
y = pdf(pd,x);
plot(x,y)
xlabel('Failure time');
ylabel('Hazard rate');
legend('2nd ABP-BP','3rd ABP-BP','4th ABP-BP','5th ABP-BP')
%Burrhazard = pdf('wbl',x,50,3,1)./(1-cdf('wbl',x,50,3,1));
%figure()
%plot(x,Burrhazard)

