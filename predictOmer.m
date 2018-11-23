% make predictions with respect to the gaussian mixture model that was fit
function [ accuracy, predict ] = predictOmer( testData,mu,Sigma,clusterWeight,classCount,clusterCount )
    %prediction
    score=zeros(size(testData,1),classCount,clusterCount);
    predict=zeros(size(testData,1),1);
    for i=1:clusterCount
        for c=1:classCount
        score(:,c,i) = gaussOmer(testData(:,1:2),mu(i,:,c),Sigma(:,:,c,i)).*clusterWeight(i,c);
        end
    end
    score=sum(score,3);
    for i=1:size(testData,1)
        predict(i)=find(score(i,:)==max(score(i,:)));
    end
    %correction of results
    result = (predict(:,1)==testData(:,3));
    accuracy = sum(double(result))/size(testData,1);
end

