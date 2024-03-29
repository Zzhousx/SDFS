function [X_new] = SDFS(X,param)
%% Notes:
% Inputs:
%   X - Original data matrix
%   param - Parameter set
% Outputs:
%   X_new - New low-dimensional data matrix 
%%%%%%%%%%%%%%%%%%%%%%%%%
% Example usage
% MATLAB
%%%%%%%%%%%%%%%%%%%%%%%%%
% clc; clear; warning off;
% load('MSRC_V1_5views.mat');
% rng('default');
% param.v = size(X,2); param.n=210; param.c = 7; param.alpha = 1; param.lambda = 1; param.gamma = 1; param.gamma2 = 1;
% [X_new] = SDFS(X,param); 
%%%%%%%%%%%
%% ===================== Parameters =====================
%%paramter:
% gamma : orthogonal constraint on W
% alpha : latent express
% lambda: l_21 norm constraint on W
alpha=param.alpha;
gamma=param.gamma;
lambda=param.lambda;
gamma2=param.gamma2;
NITER = 100; % Maximum number of iterations
n = param.n; % Number of samples
v = param.v; % Number of views
c = param.c; % Number of classes
select=0.15 ; % The proportion of the selected feature.
c1=2*c;
pn=15;
islocal=1;
%%%%%%%%%%%%%%%
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';
options.t = 10;
%% ===================== Normalize  =====================
temp_std=cell(1,v);
XX=[];
for vv=1:v
    temp_std{vv}=std(X{1,vv},0,2);
    for i=1:size(X{1,vv},1)
          meanvalue_fea=mean(X{1,vv}(i,:));
          X{1,vv}(i,:)=(X{1,vv}(i,:)-meanvalue_fea)/temp_std{vv}(i,:);
    end
    X{vv}=X{vv}';
    XX=[XX;X{1,vv}];  
    X{vv}=X{vv}';
end
%% ===================== initialize =====================
W=cell(1,v); %feature selection matrix
H=cell(1,v); 
F=cell(1,v); 
Xnor=cell(1,v);
q=cell(1,v);
Q=cell(1,v);
S=cell(1,v);
I=eye(c1);
Scon = zeros(n);
w1 = ones(1,v)/v;
idxx = cell(1,v);
ed = cell(1,v);
for vIndex=1:v
    dv = size(X{vIndex},2);
    Q{vIndex} = eye(dv);
    W{vIndex}=ones(dv,c1);
    H{vIndex}=ones(c1,c);
    F{vIndex}=rand(n,c);
    Xnor{vIndex}=X{vIndex};
    TempNorData=Xnor{vIndex}';
    % initialize S
    [S{vIndex}, ~] = InitializeSIGs(TempNorData,TempNorData, pn, 0);
    % construct graph 
    Av = constructW(TempNorData',options);
    A{vIndex}=Av;
    Scon = Scon + S{vIndex};
    S{vIndex} = (S{vIndex}+S{vIndex}')/2;
    D{vIndex} = diag(sum(S{vIndex}));
end
% initialize Scon
Scon = Scon/v;
for j = 1:n
    Scon(j,:) = Scon(j,:)/sum(Scon(j,:));
end
Scon = (Scon+Scon')/2;
Dcon = diag(sum(Scon));
%% ===================== updating =====================
for iter = 1:NITER
    for iterv = 1:v
       F{iterv} = F{iterv}.*((S{iterv}*Xnor{iterv}*W{iterv}*H{iterv}+2*alpha*A{iterv}'*Scon*F{iterv}+gamma2*F{iterv})./(D{iterv}*F{iterv}+alpha*F{iterv}*F{iterv}'*Dcon*F{iterv}+alpha*Dcon*F{iterv}*F{iterv}'*F{iterv}+gamma2*F{iterv}*F{iterv}'*F{iterv}+eps));
       W{iterv} = W{iterv}.*((Xnor{iterv}'*S{iterv}'*F{iterv}*H{iterv}'+gamma*W{iterv})./(Xnor{iterv}'*D{iterv}*Xnor{iterv}*W{iterv}*H{iterv}*H{iterv}'+lambda*Q{iterv}*W{iterv}+gamma*W{iterv}*W{iterv}'*W{iterv}+eps));
       H{iterv} = H{iterv}.*((2*W{iterv}'*Xnor{iterv}'*S{iterv}'*F{iterv})./(2*W{iterv}'*Xnor{iterv}'*D{iterv}*Xnor{iterv}*W{iterv}*H{iterv}+eps));
       qj{iterv} = sqrt(sum(W{iterv}.*W{iterv},2)+eps);
       q{iterv} = 0.5./qj{iterv};
       Q{iterv} = diag(q{iterv});
        ed{iterv} = L2_distance_1(Xnor{iterv}', Xnor{iterv}');
        [~, idxx{iterv}] = sort(ed{iterv}, 2); 
        S{iterv} = zeros(n);
        for i = 1:n
            id = idxx{iterv}(i,2:pn+2);
            di = ed{iterv}(i, id);
            numerator = di(pn+1)-di+2*w1(iterv)*Scon(i,id(:))-2*w1(iterv)*Scon(i,id(pn+1));
            denominator1 = pn*di(pn+1)-sum(di(1:pn));
            denominator2 = 2*w1(iterv)*sum(Scon(i,id(1:pn)))-2*pn*w1(iterv)*Scon(i,id(pn+1));
            S{iterv}(i,id) = max(numerator/(denominator1+denominator2+eps),0);
        end
        S{iterv} = (S{iterv}+S{iterv}')/2;
        D{iterv}=diag(sum(S{iterv}));
        US = Scon - S{iterv};
        distUS = norm(US, 'fro')^2;
        if distUS == 0
            distUS = eps;
        end
        w1(iterv) = 0.5/sqrt(distUS);
    end
     for iterv = 1:v
        dist{iterv} = L2_distance_1(F{iterv}',F{iterv}');
     end
    Scon = zeros(n);
    for i=1:n
        idx = zeros();
        for iterv = 1:v
            s0 = S{iterv}(i,:);
            idx = [idx,find(s0>0)];
        end
        idxs = unique(idx(2:end));
        if islocal == 1
            idxs0 = idxs;
        else
            idxs0 = 1:n;
        end
        for iterv = 1:v
            s1 = S{iterv}(i,:);
            si = s1(idxs0);
            di2{iterv} = dist{iterv}(i,idxs0);
            mw = v*w1(iterv);
            lmw = alpha/mw;
            q11(iterv,:) = si-0.5*lmw*di2{iterv};
        end
        Scon(i,idxs0) = updateS(q11,v);
        clear q11;
    end
    Scon = (Scon+Scon')/2;
    Dcon = diag(sum(Scon));


 %% Update alpha
%     gvsum=0;
%     for objIndex=1:v
%         gv=trace(A{objIndex}'*Dcon*A{objIndex})-2*trace(A{objIndex}'*Scon*F{objIndex}*F{objIndex}')+trace(F{objIndex}*F{objIndex}'*Dcon*F{objIndex}*F{objIndex}');;
%         gvsum=gvsum+gv.^(1/(1-r1));
%     end
%     for objIndex=1:v
%         gtemp=trace(A{objIndex}'*Dcon*A{objIndex})-2*trace(A{objIndex}'*Scon*F{objIndex}*F{objIndex}')+trace(F{objIndex}*F{objIndex}'*Dcon*F{objIndex}*F{objIndex}');
%         gtemp=gtemp.^(1/(1-r1));
%         alpha(objIndex)=gtemp/gvsum;
%     end
% ===================== calculate obj =====================
    NormX=0;
    obj1=0;
    q1=0;
    q2=0;
    q3=0;
    for objIndex=1:v
        Term1 = trace(F{objIndex}'*D{objIndex}*F{objIndex})-2*trace(F{objIndex}'*S{objIndex}*Xnor{objIndex}*W{objIndex}*H{objIndex})+trace(H{objIndex}'*W{objIndex}'*Xnor{objIndex}'*D{objIndex}*Xnor{objIndex}*W{objIndex}*H{objIndex});
        q1=q1+trace(F{objIndex}'*D{objIndex}*F{objIndex});
        q2=q2+trace(F{objIndex}'*S{objIndex}*Xnor{objIndex}*W{objIndex}*H{objIndex});
        q3=q3+trace(H{objIndex}'*W{objIndex}'*Xnor{objIndex}'*D{objIndex}*Xnor{objIndex}*W{objIndex}*H{objIndex});
        Term2 = trace(A{objIndex}'*Dcon*A{objIndex})-2*trace(A{objIndex}'*Scon*F{objIndex}*F{objIndex}')+trace(F{objIndex}*F{objIndex}'*Dcon*F{objIndex}*F{objIndex}');
        Term3 = sum(qj{objIndex});
        Term4 = trace((W{objIndex}'*W{objIndex}-I)*(W{objIndex}'*W{objIndex}-I)');
        Term5 = norm(Scon-S{objIndex},'fro').^2;
        tempobj(objIndex)=Term1+alpha*Term2+lambda*Term3+gamma*Term4+w1(objIndex)*Term5;
        
        NormX = NormX + norm(Xnor{objIndex},'fro')^2;
    end

    r=3;
    HH = bsxfun(@power,tempobj, 1/(1-r));
    alpha1 = bsxfun(@rdivide,HH,sum(HH));
    alpha_r = alpha1.^r;
    obj1(iter) = alpha_r*tempobj';
    obj(iter) = (alpha_r*tempobj')/NormX;
    if iter == 1
        err = 0;
    else
        err = obj(iter)-obj(iter-1);
    end
    fprintf('iteration =  %d:  obj: %.10f; err: %.8f  \n', ...
        iter, obj(iter), err);
    if (abs(err))<1e-7 
        if iter > 10
            break;
        end
    end
%%
for i = 1:v
    W{i} = W{i}';
end
Wcon = DataConcatenate(W);
Wcon = Wcon';
for i = 1:v
    W{i} = W{i}';
end
d = size(XX,1);
selectedFeas = ceil(select*d);
w = [];
for i = 1:d
    w = [w norm(Wcon(i,:),2)];
end
[~,index] = sort(w,'descend');
X_new = XX(index(1:selectedFeas),:);
end





