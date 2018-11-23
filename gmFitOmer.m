% gaussian mixture model fitting
% PARAMETERS 
% cluster count: number of clusters you want to fit
% covType: covariance matrix type of the gaussians
% OUTPUTS
% mu: mu's of gaussians
% Sigma: sigma of the gaussians
% clusterWeight: cluster weights for the specific gaussian
% likelihood: likelihood of the fit
function [ mu, Sigma, clusterWeight, likelihood ] = gmFitOmer( data,clusterCount ,covType )
    %setting parameters
    clusterWeight=ones(clusterCount,1);
    clusterWeight= clusterWeight./size(clusterWeight,1);
    mu = [20 20];
    mu=repmat(mu,clusterCount,1);
    Sigma=zeros(2,2,clusterCount);
    for i=1:clusterCount
        Sigma(:,:,i) = [20 0; 0 20];
    end
    gauss=zeros(size(data,1),clusterCount);
    F=zeros(size(data,1),clusterCount);
    mc=zeros(clusterCount,1);
    for i=1:1000
        prewMu=mu;
        %E step
        for c=1:clusterCount
            gauss(:,c)=gaussOmer(data,mu(c,:),Sigma(:,:,c));
            F(:,c)=clusterWeight(c)*gauss(:,c);
        end
        r=F./sum(F,2);
        %M step
        for c=1:clusterCount
            mc(c) = sum(r(:,c));
            clusterWeight(c)=mc(c)/sum(mc);
            mu(c,:)=r(:,c)'*data/mc(c);
            Xm = bsxfun(@minus, data, mu(c,:));     %mean - data
            w=bsxfun(@times, Xm, r(:,c));           %weighting components     
            %spherical
            if(covType==1)
                d=zeros(1,size(r,1));
                for j = 1: size(Xm,1)
                   d(j)= power(norm(Xm(j,:)),2);
                end
                s=((d*r(:,c))/mc(c))/size(data,2);
                Sigma(:,:,c)= [s 0 ; 0 s];
            %diagonal
            elseif(covType==2)
               sig1=(power(Xm(:,1),2)'*r(:,c))/mc(c);
               sig2=(power(Xm(:,2),2)'*r(:,c))/mc(c);
               Sigma(:,:,c)= [sig1 0 ; 0 sig2];
            %any
            else
                Sigma(:,:,c) = (w)'*((Xm))./mc(c);
            end 
            l(:,c)=(gaussOmer(data, mu(c,:),Sigma(:,:,c)).*clusterWeight(c));
        end
        likelihood=sum(log(sum(l,2)));
        %break the loop when mu is almost same as previous iteration 
         if(abs(mu-prewMu)<10e-15)
             break;
         end
    end
end

