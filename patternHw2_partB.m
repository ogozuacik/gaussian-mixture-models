clear; clc;
data=importdata('cs551_hw2_q1.dat');

%%%%%%%%%%PART B %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%OMER %%%%%%%%%%%%%%%%%%%

data1=data(1:1000,:);
data2=data(1001:2000,:);
data3=data(2001:3000,:);
clear data;

%randomizing the order of data for classes 1,2&3 
data1=data1(randperm(size(data1,1)),:);
data2=data2(randperm(size(data2,1)),:);
data3=data3(randperm(size(data3,1)),:);

%selecting first 500 from each for train and remaining 500 for test
trainData= [data1(1:500,:) ; data2(1:500,:) ; data3(1:500,:)];
testData= [data1(501:1000,:) ; data2(501:1000,:) ; data3(501:1000,:)];
clear data1; clear data2; clear data3;

%generating classifiers with different bin sizes and
%testing them with train and test data to prevent overfitting
testAccuracy=zeros(90,1);
trainAccuracy=zeros(90,1);
for i=11:101
    %bincount per axis is tested for different values
    bincount= i; % -1 
    edges = linspace(0,50,bincount);
    classify=histClassiffierOmer(trainData, edges);
    [ testAccuracy(i-10) , ~ ] = predictHistOmer( classify, testData , edges);
    [ trainAccuracy(i-10), ~ ] = predictHistOmer( classify, trainData , edges );
end
%plotting test&train accuracy vs bincount
figure()
plot(10:100,trainAccuracy, 'DisplayName','Train Accuracy') ;hold on;
plot(10:100,testAccuracy, 'DisplayName','Test Accuracy') ;hold on;
legend('show')
title('Test & Train Accuracy vs Bincount per Axis')
xlabel('Bincount'); ylabel('Accuracy');

%finding best test accuracy depending on bincount
bincount=find(testAccuracy==max(testAccuracy)) + 10;
edges = linspace(0,50,bincount);
classify=histClassiffierOmer(trainData, edges);

%plotting histogram for best test accuracy
figure()
histogram2(trainData(1:500,1),trainData(1:500,2),edges,edges); hold on;
histogram2(trainData(501:1000,1),trainData(501:1000,2),edges,edges); hold on;
histogram2(trainData(1001:1500,1),trainData(1001:1500,2),edges,edges);
%plotting classification matrix for best test accuracy
figure()
imagesc(imrotate(classify,90));

%making predictions with respect to clasify matrix with best accuracy 
%to test data and trainData and giving their results
[testAccuracy, testResult ] = predictHistOmer( classify, testData , edges);
[trainAccuracy, trainResult ] = predictHistOmer( classify, trainData , edges );

%plotting confusion matrix
Conf_Mat_train = confusionmat(trainData(:,3),trainResult(:,1))
Conf_Mat_test = confusionmat(testData(:,3),testResult(:,1))

%c=table(Conf_Mat_test,'RowNames',{'0','1','2','3'})
%'RowNames',LastName
%'ColumnNames',LastName
