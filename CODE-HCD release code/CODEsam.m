function [c]=CODEsam(x1,x2,Cdl)

[x, y, z] = size(x1);
% Cdl=zeros(x,y);
num = sum(x1 .* x2, 3);
den = sqrt(sum(x1.^2, 3) .* sum(x2.^2, 3));
% num1=num(:); den1=den(:);
% index=find(den1==0);
% num1(index)=[]; den1(index)=[];
Z = acosd(num ./ den);
% Z = Z/max(Z(:));
% Z = x1-x2;
Z = reshape(Z,x*y,[])';
% initialize
mu = 0.001;
lambda = 10; %Wetland
iter = 500;
[c, d1, d2, d3, d4] = deal(zeros(x*y,1));
cdl = double(Cdl(:));
temp = Z.*Z;
c1left = sqrt(1./(temp+mu*ones(1,x*y)))';
psi = Z'.*Z';

for i=1:iter
    
    % update c1
    c1 = mu*(c1left.*(c-d1));
%     c1 = mu*(c1left*(c-d1));

    % update c2
    c2 = psi/mu+c-d2;

    % update c3
    c3 = (lambda*cdl+mu*(c-d3))/(lambda+mu);

    % update c4
    c4 = c-d4;
    c4(c4<0) = 0;
    c4(c4>1) = 1;

    % update c
    c = (c1+c2+c3+c4+d1+d2+d3+d4)/4;

    % update d
    d1 = d1+c1-c;
    d2 = d2+c2-c;
    d3 = d3+c3-c;
    d4 = d4+c4-c;

end
end