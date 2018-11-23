clear; clc;
data=importdata('cs551_hw2_q1.dat');

%%%%%%%%%%PART A %%%%%%%%%%%%%%%%%%%
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

% YOU CAN CHANGE VARIABLES IN HERE FOR DIFFERENT FIT'S FOR THE DATA
%training system class by class
clusterCount=3;             %cluster count per class
classCount=3;
covType = 1; %1 spherical %2 diagonal %3 arbitrary

clusterWeight=zeros(clusterCount,classCount);
mu=zeros(clusterCount,2,classCount);
Sigma =zeros(2,2,classCount,clusterCount);
likelihood=zeros(classCount,1);

%1 training for class 1
[mu(:,:,1), Sigma(:,:,1,:), clusterWeight(:,1), likelihood(1)] = gmFitOmer(trainData(1:500,1:2),clusterCount,covType);
%2 training for class 2
[mu(:,:,2), Sigma(:,:,2,:), clusterWeight(:,2), likelihood(2)] = gmFitOmer(trainData(501:1000,1:2),clusterCount,covType);
%3 training for class 3
[mu(:,:,3), Sigma(:,:,3,:), clusterWeight(:,3), likelihood(3)] = gmFitOmer(trainData(1001:1500,1:2),clusterCount,covType);

%plotting
figure
plot(trainData(1:500,1),trainData(1:500,2), 'ob', 'DisplayName','class1'); hold on 
plot(trainData(501:1000,1),trainData(501:1000,2), '+r', 'DisplayName','class2'); hold on 
plot(trainData(1001:1500,1),trainData(1001:1500,2), '*g', 'DisplayName','class3'); hold on 

%plotting cluster centers as large x's

for i=1:clusterCount
    for j=1:classCount
        plot(mu(i,1,j),mu(i,2,j),'xk','MarkerSize',20); hold on
    end
end
%plotting contours
bincount = 40;
x = linspace(0, 50, bincount);
[X, Y] = meshgrid(x, x);
gridX = [X(:), Y(:)];

for c=1:classCount
    for i=1:clusterCount
        n = gaussOmer(gridX, mu(i, :, c), Sigma(:,:,c,i))*clusterWeight(i,c);
        N = reshape(n, bincount, bincount);
        contour(x, x, N);
    end
end
axis tight

%prediction
[testAccuracy, testResult] = predictOmer(testData,mu,Sigma,clusterWeight,classCount,clusterCount);
[trainAccuracy, trainResult]  = predictOmer(trainData,mu,Sigma,clusterWeight,classCount,clusterCount);

disp('Test Accuracy'); disp(testAccuracy);
disp('Train Accuracy'); disp(trainAccuracy);

disp('Confusion matrix for train data')
Conf_Mat_train = confusionmat(trainData(:,3),trainResult(:,1))
disp('Confusion matrix for test data')
Conf_Mat_test = confusionmat(testData(:,3),testResult(:,1))

