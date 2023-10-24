function [W,F,S,A,Scon,obj,H,w1] = SDFS(X,param,options)
%%paramter:
% gamma: orthogonal constraint on W
% alpha: latent express
% lambda: l_21 norm constraint on W
%% ===================== Parameters =====================
alpha=param.alpha;
gamma=param.gamma;
lambda=param.lambda;
gamma2=param.gamma2;
NITER = param.NITER;
n = param.n;
v = param.v;
c = param.c;
c1=c;
pn=15;
islocal=1;
%% ===================== initialize =====================
W=cell(1,v);
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
 %% update F
        F{iterv} = F{iterv}.*((S{iterv}*Xnor{iterv}*W{iterv}*H{iterv}+2*alpha*A{iterv}'*Scon*F{iterv}+gamma2*F{iterv})./(D{iterv}*F{iterv}+alpha*F{iterv}*F{iterv}'*Dcon*F{iterv}+alpha*Dcon*F{iterv}*F{iterv}'*F{iterv}+gamma2*F{iterv}*F{iterv}'*F{iterv}+eps));
 %% update W
        W{iterv} = W{iterv}.*((Xnor{iterv}'*S{iterv}'*F{iterv}*H{iterv}'+gamma*W{iterv})./(Xnor{iterv}'*D{iterv}*Xnor{iterv}*W{iterv}*H{iterv}*H{iterv}'+lambda*Q{iterv}*W{iterv}+gamma*W{iterv}*W{iterv}'*W{iterv}+eps));
 %% update H
        H{iterv} = H{iterv}.*((2*W{iterv}'*Xnor{iterv}'*S{iterv}'*F{iterv})./(2*W{iterv}'*Xnor{iterv}'*D{iterv}*Xnor{iterv}*W{iterv}*H{iterv}+eps));
 %% construct l_21 norm matrix  
        qj{iterv} = sqrt(sum(W{iterv}.*W{iterv},2)+eps);
        q{iterv} = 0.5./qj{iterv};
        Q{iterv} = diag(q{iterv});
 %% update S
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
 %% update w1
        US = Scon - S{iterv};
        distUS = norm(US, 'fro')^2;
        if distUS == 0
            distUS = eps;
        end
        w1(iterv) = 0.5/sqrt(distUS);
    end
 %% update Scon
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
    obj2=0;
    obj3=0;
    obj4=0;
    obj5=0;
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










