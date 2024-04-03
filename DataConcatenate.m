function [XX, n, nfeat] = DataConcatenate(D)
v = size(D,2);
for i = 1:v
     if i == 1
         XX = D{i};     
     else 
         XX = cat(2, XX, D{i});
     end
end
[n,nfeat] = size(XX);
% disp(['DataConcatenate.n = ',num2str(n)]);
% disp(['DataConcatenate.nfeat = ',num2str(nfeat)]);