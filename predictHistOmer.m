function [ accuracy, prediction ] = predictHistOmer( classify, testData, edges )
    prediction=zeros(size(testData,1),1);
    for i=1:size(testData,1)
       temp=histcounts2(testData(i,1),testData(i,2),edges,edges);
       [row,col]= find(temp==1);
       prediction(i) = classify(row,col);
    end
    %accuracy of results
    result = (prediction(:,1)==testData(:,3));
    accuracy = sum(double(result))/size(testData,1);
end

