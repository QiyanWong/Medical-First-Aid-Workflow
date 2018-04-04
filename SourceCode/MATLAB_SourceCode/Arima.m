M1 = xlsread('duration.xls','Sheet1','C2:C3919');
data0=M1(15:20,3);
Mdl = arima(1,1,1);
EstMdl = estimate(Mdl,data0);
con0 = EstMdl.Constant;
ar0 = EstMdl.AR{1};
ma0 = EstMdl.MA{1};
var0 = EstMdl.Variance;

[Y,YMSE] = forecast(EstMdl,6,'Y0',data0);

lower = Y - 1.96*sqrt(YMSE);
upper = Y + 1.96*sqrt(YMSE);

figure
plot(data0,'Color',[.7,.7,.7]);
hold on
h1 = plot(5:10,lower,'r:','LineWidth',2);
plot(5:10,upper,'r:','LineWidth',2)
h2 = plot(5:10,Y,'k','LineWidth',2);
legend([h1 h2],'95% Interval','Forecast',...
	     'Location','NorthWest')
title('Activity Index Forecast')
hold off
sum = 0;
iter = 1;
count = 0;

for i=1:length(M1)
    count = count+1;
    data0 = M1(iter:iter+5,3); 
    Mdl = arima(1,1,1);
    EstMdl = estimate(Mdl,data0);
    con0 = EstMdl.Constant;
    ar0 = EstMdl.AR{1};
    ma0 = EstMdl.MA{1};
    var0 = EstMdl.Variance;
    [Y,YMSE] = forecast(EstMdl,6,'Y0',data0);
    if(iter > 3900) break;
    end
    iter = iter +7;
    result = abs(Y(1)-M1(iter+1,3))
    error(count) = result;
    sum = sum + result;
end
sum/count
S = std(error)