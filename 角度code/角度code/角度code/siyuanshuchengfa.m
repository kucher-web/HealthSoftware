function [normalizedMatrixQ] = siyuanshuchengfa(p,q)
%SIYUANSHUCHENGFA 此处显示有关此函数的摘要
%   此处显示详细说明
    Q=[p(4),-p(3),p(2),p(1);p(3),p(4),-p(1),p(2);-p(2),p(1),p(4),p(3);-p(1),-p(2),-p(3),p(4)]*q';
    Norm = norm(Q);
    normalizedMatrixQ = Q ./ Norm;
    if normalizedMatrixQ (4)<0
     normalizedMatrixQ=-normalizedMatrixQ;
    end
end

