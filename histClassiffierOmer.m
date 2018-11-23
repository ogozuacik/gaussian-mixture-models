%making prediction with respect to the estimation from the histogram 
function [classify] = histClassiffierOmer( trainData, edges )
 
    N1 = histcounts2(trainData(1:500,1), trainData(1:500,2),edges,edges);
    N2 = histcounts2(trainData(501:1000,1), trainData(501:1000,2),edges,edges);
    N3 = histcounts2(trainData(1001:1500,1), trainData(1001:1500,2),edges,edges);

    classify=zeros(size(N1));

    %generating a classifier from histograms
    for i=1:size(N1,1)
        for j=1:size(N1,1)
            if(N1(i,j) > N2(i,j) && N1(i,j) > N3(i,j))
                classify(i,j) = 1;
            elseif (N2(i,j) > N1(i,j) && N2(i,j) > N3(i,j))
                classify(i,j) = 2;
            elseif(N3(i,j) > N1(i,j) && N2(i,j) < N3(i,j))
                classify(i,j) = 3;
            else
                classify(i,j) = 0; %nearest neighbours can be checked for improved accuracy
            end
        end
    end
end

