function [W] = SDFS(X,param)
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
for vv=1:v
    temp_std{vv}=std(X{1,vv},0,2);
    for i=1:size(X{1,vv},1)
          meanvalue_fea=mean(X{1,vv}(i,:));
          X{1,vv}(i,:)=(X{1,vv}(i,:)-meanvalue_fea)/temp_std{vv}(i,:);
    end
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
end

function [S, D] = InitializeSIGs(X,X1, k, issymmetric)
if nargin < 3
    issymmetric = 1;
end
if nargin < 2
    k = 5;
end
[~, n] = size(X);
D = L2_distance_1(X, X1);
[~, idx] = sort(D, 2); 
S = zeros(n);
for i = 1:n
    id = idx(i,2:k+2);
    di = D(i, id);
    S(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end
if issymmetric == 1
    S = (S+S')/2;
end


function W = constructW(fea,options)
bSpeed  = 1;

if (~exist('options','var'))
   options = [];
end

if isfield(options,'Metric')
    warning('This function has been changed and the Metric is no longer be supported');
end


if ~isfield(options,'bNormalized')
    options.bNormalized = 0;
end

%=================================================
if ~isfield(options,'NeighborMode')
    options.NeighborMode = 'KNN';
end

switch lower(options.NeighborMode)
    case {lower('KNN')}  
        if ~isfield(options,'k')
            options.k = 5;
        end
    case {lower('Supervised')}
        if ~isfield(options,'bLDA')
            options.bLDA = 0;
        end
        if options.bLDA
            options.bSelfConnected = 1;
        end
        if ~isfield(options,'k')
            options.k = 0;
        end
        if ~isfield(options,'gnd')
            error('Label(gnd) should be provided under ''Supervised'' NeighborMode!');
        end
        if ~isempty(fea) && length(options.gnd) ~= size(fea,1)
            error('gnd doesn''t match with fea!');
        end
    otherwise
        error('NeighborMode does not exist!');
end

%=================================================

if ~isfield(options,'WeightMode')
    options.WeightMode = 'HeatKernel';
end

bBinary = 0;
bCosine = 0;
switch lower(options.WeightMode)
    case {lower('Binary')}
        bBinary = 1; 
    case {lower('HeatKernel')}
        if ~isfield(options,'t')
            nSmp = size(fea,1);
            if nSmp > 3000
                D = EuDist2(fea(randsample(nSmp,3000),:));
            else
                D = EuDist2(fea);
            end
            options.t = mean(mean(D));
        end
    case {lower('Cosine')}
        bCosine = 1;
    otherwise
        error('WeightMode does not exist!');
end

%=================================================

if ~isfield(options,'bSelfConnected')
    options.bSelfConnected = 0;
end

%=================================================

if isfield(options,'gnd') 
    nSmp = length(options.gnd);
else
    nSmp = size(fea,1);
end
maxM = 62500000; 
BlockSize = floor(maxM/(nSmp*3));
if strcmpi(options.NeighborMode,'Supervised')
    Label = unique(options.gnd);
    nLabel = length(Label);
    if options.bLDA
        G = zeros(nSmp,nSmp);
        for idx=1:nLabel
            classIdx = options.gnd==Label(idx);
            G(classIdx,classIdx) = 1/sum(classIdx);
        end
        W = sparse(G);
        return;
    end
    
    switch lower(options.WeightMode)
        case {lower('Binary')}
            if options.k > 0
                G = zeros(nSmp*(options.k+1),3);
                idNow = 0;
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    D = EuDist2(fea(classIdx,:),[],0);
                    [dump idx] = sort(D,2); % sort each row
                    clear D dump;
                    idx = idx(:,1:options.k+1);
                    
                    nSmpClass = length(classIdx)*(options.k+1);
                    G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[options.k+1,1]);
                    G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:));
                    G(idNow+1:nSmpClass+idNow,3) = 1;
                    idNow = idNow+nSmpClass;
                    clear idx
                end
                G = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
                G = max(G,G');
            else
                G = zeros(nSmp,nSmp);
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    G(classIdx,classIdx) = 1;
                end
            end
            
            if ~options.bSelfConnected
                for i=1:size(G,1)
                    G(i,i) = 0;
                end
            end
            
            W = sparse(G);
        case {lower('HeatKernel')}
            if options.k > 0
                G = zeros(nSmp*(options.k+1),3);
                idNow = 0;
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    D = EuDist2(fea(classIdx,:),[],0);
                    [dump idx] = sort(D,2); 
                    clear D;
                    idx = idx(:,1:options.k+1);
                    dump = dump(:,1:options.k+1);
                    dump = exp(-dump/(2*options.t^2));
                    
                    nSmpClass = length(classIdx)*(options.k+1);
                    G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[options.k+1,1]);
                    G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:));
                    G(idNow+1:nSmpClass+idNow,3) = dump(:);
                    idNow = idNow+nSmpClass;
                    clear dump idx
                end
                G = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
            else
                G = zeros(nSmp,nSmp);
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    D = EuDist2(fea(classIdx,:),[],0);
                    D = exp(-D/(2*options.t^2));
                    G(classIdx,classIdx) = D;
                end
            end
            
            if ~options.bSelfConnected
                for i=1:size(G,1)
                    G(i,i) = 0;
                end
            end

            W = sparse(max(G,G'));
        case {lower('Cosine')}
            if ~options.bNormalized
                fea = NormalizeFea(fea);
            end

            if options.k > 0
                G = zeros(nSmp*(options.k+1),3);
                idNow = 0;
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    D = fea(classIdx,:)*fea(classIdx,:)';
                    [dump idx] = sort(-D,2); 
                    clear D;
                    idx = idx(:,1:options.k+1);
                    dump = -dump(:,1:options.k+1);
                    
                    nSmpClass = length(classIdx)*(options.k+1);
                    G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[options.k+1,1]);
                    G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:));
                    G(idNow+1:nSmpClass+idNow,3) = dump(:);
                    idNow = idNow+nSmpClass;
                    clear dump idx
                end
                G = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
            else
                G = zeros(nSmp,nSmp);
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    G(classIdx,classIdx) = fea(classIdx,:)*fea(classIdx,:)';
                end
            end

            if ~options.bSelfConnected
                for i=1:size(G,1)
                    G(i,i) = 0;
                end
            end

            W = sparse(max(G,G'));
        otherwise
            error('WeightMode does not exist!');
    end
    return;
end
if bCosine && ~options.bNormalized
    Normfea = NormalizeFea(fea);
end
if strcmpi(options.NeighborMode,'KNN') && (options.k > 0)
    if ~(bCosine && options.bNormalized)
        G = zeros(nSmp*(options.k+1),3);
        for i = 1:ceil(nSmp/BlockSize)
            if i == ceil(nSmp/BlockSize)
                smpIdx = (i-1)*BlockSize+1:nSmp;
                dist = EuDist2(fea(smpIdx,:),fea,0);

                if bSpeed
                    nSmpNow = length(smpIdx);
                    dump = zeros(nSmpNow,options.k+1);
                    idx = dump;
                    for j = 1:options.k+1
                        [dump(:,j),idx(:,j)] = min(dist,[],2);
                        temp = (idx(:,j)-1)*nSmpNow+[1:nSmpNow]';
                        dist(temp) = 1e100;
                    end
                else
                    [dump idx] = sort(dist,2); % sort each row
                    idx = idx(:,1:options.k+1);
                    dump = dump(:,1:options.k+1);
                end
                
                if ~bBinary
                    if bCosine
                        dist = Normfea(smpIdx,:)*Normfea';
                        dist = full(dist);
                        linidx = [1:size(idx,1)]';
                        dump = dist(sub2ind(size(dist),linidx(:,ones(1,size(idx,2))),idx));
                    else
                        dump = exp(-dump/(2*options.t^2));
                    end
                end
                
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),2) = idx(:);
                if ~bBinary
                    G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),3) = dump(:);
                else
                    G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),3) = 1;
                end
            else
                smpIdx = (i-1)*BlockSize+1:i*BlockSize;
            
                dist = EuDist2(fea(smpIdx,:),fea,0);
                
                if bSpeed
                    nSmpNow = length(smpIdx);
                    dump = zeros(nSmpNow,options.k+1);
                    idx = dump;
                    for j = 1:options.k+1
                        [dump(:,j),idx(:,j)] = min(dist,[],2);
                        temp = (idx(:,j)-1)*nSmpNow+[1:nSmpNow]';
                        dist(temp) = 1e100;
                    end
                else
                    [dump idx] = sort(dist,2); 
                    idx = idx(:,1:options.k+1);
                    dump = dump(:,1:options.k+1);
                end
                
                if ~bBinary
                    if bCosine
                        dist = Normfea(smpIdx,:)*Normfea';
                        dist = full(dist);
                        linidx = [1:size(idx,1)]';
                        dump = dist(sub2ind(size(dist),linidx(:,ones(1,size(idx,2))),idx));
                    else
                        dump = exp(-dump/(2*options.t^2));
                    end
                end
                
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),2) = idx(:);
                if ~bBinary
                    G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),3) = dump(:);
                else
                    G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),3) = 1;
                end
            end
        end

        W = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
    else
        G = zeros(nSmp*(options.k+1),3);
        for i = 1:ceil(nSmp/BlockSize)
            if i == ceil(nSmp/BlockSize)
                smpIdx = (i-1)*BlockSize+1:nSmp;
                dist = fea(smpIdx,:)*fea';
                dist = full(dist);

                if bSpeed
                    nSmpNow = length(smpIdx);
                    dump = zeros(nSmpNow,options.k+1);
                    idx = dump;
                    for j = 1:options.k+1
                        [dump(:,j),idx(:,j)] = max(dist,[],2);
                        temp = (idx(:,j)-1)*nSmpNow+[1:nSmpNow]';
                        dist(temp) = 0;
                    end
                else
                    [dump idx] = sort(-dist,2); 
                    idx = idx(:,1:options.k+1);
                    dump = -dump(:,1:options.k+1);
                end

                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),2) = idx(:);
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),3) = dump(:);
            else
                smpIdx = (i-1)*BlockSize+1:i*BlockSize;
                dist = fea(smpIdx,:)*fea';
                dist = full(dist);
                
                if bSpeed
                    nSmpNow = length(smpIdx);
                    dump = zeros(nSmpNow,options.k+1);
                    idx = dump;
                    for j = 1:options.k+1
                        [dump(:,j),idx(:,j)] = max(dist,[],2);
                        temp = (idx(:,j)-1)*nSmpNow+[1:nSmpNow]';
                        dist(temp) = 0;
                    end
                else
                    [dump idx] = sort(-dist,2); 
                    idx = idx(:,1:options.k+1);
                    dump = -dump(:,1:options.k+1);
                end

                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),2) = idx(:);
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),3) = dump(:);
            end
        end

        W = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
    end
    
    if bBinary
        W(logical(W)) = 1;
    end
    
    if isfield(options,'bSemiSupervised') && options.bSemiSupervised
        tmpgnd = options.gnd(options.semiSplit);
        
        Label = unique(tmpgnd);
        nLabel = length(Label);
        G = zeros(sum(options.semiSplit),sum(options.semiSplit));
        for idx=1:nLabel
            classIdx = tmpgnd==Label(idx);
            G(classIdx,classIdx) = 1;
        end
        Wsup = sparse(G);
        if ~isfield(options,'SameCategoryWeight')
            options.SameCategoryWeight = 1;
        end
        W(options.semiSplit,options.semiSplit) = (Wsup>0)*options.SameCategoryWeight;
    end
    
    if ~options.bSelfConnected
        W = W - diag(diag(W));
    end

    if isfield(options,'bTrueKNN') && options.bTrueKNN
        
    else
        W = max(W,W');
    end
    
    return;
end
switch lower(options.WeightMode)
    case {lower('Binary')}
        error('Binary weight can not be used for complete graph!');
    case {lower('HeatKernel')}
        W = EuDist2(fea,[],0);
        W = exp(-W/(2*options.t^2));
    case {lower('Cosine')}
        W = full(Normfea*Normfea');
    otherwise
        error('WeightMode does not exist!');
end

if ~options.bSelfConnected
    for i=1:size(W,1)
        W(i,i) = 0;
    end
end
W = max(W,W');
function d = L2_distance_1(a,b)
if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))]; 
  b = [b; zeros(1,size(b,2))]; 
end

aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;

d = real(d);
d = max(d,0);
d = d - diag(diag(d)); 
function [x, ft] = updateS(q0, m)
if nargin < 2
    m = 1;
end
ft=1;
n = length(q0);
p0 = sum(q0,1)/m-mean(sum(q0,1))/m + 1/n;
vmin = min(p0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = lambda_m-p0;
        posidx = v1>0;
        npos = sum(posidx);
        g = npos/n-1;
        if 0 == g
            g = eps;
        end
        f = sum(v1(posidx))/n - lambda_m;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(-v1,0);
            break;
        end
    end
    x = max(-v1,0);
else
    x = p0;
end

function D = EuDist2(fea_a,fea_b,bSqrt)
if ~exist('bSqrt','var')
    bSqrt = 1;
end

if (~exist('fea_b','var')) || isempty(fea_b)
    aa = sum(fea_a.*fea_a,2);
    ab = fea_a*fea_a';
    
    if issparse(aa)
        aa = full(aa);
    end
    
    D = bsxfun(@plus,aa,aa') - 2*ab;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
    D = max(D,D');
else
    aa = sum(fea_a.*fea_a,2);
    bb = sum(fea_b.*fea_b,2);
    ab = fea_a*fea_b';

    if issparse(aa)
        aa = full(aa);
        bb = full(bb);
    end

    D = bsxfun(@plus,aa,bb') - 2*ab;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
end









