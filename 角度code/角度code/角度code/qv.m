fid=fopen('12.12.1.txt','r');%TMZ067 TMZ068
tline=fgetl(fid);%读取第一行
data1={};
while ischar(tline)
    c=strsplit(tline)
    data1=[data1;c];
    tline=fgetl(fid);
end
fclose(fid);
data1=cellfun(@str2double,data1,'UniformOutput',false);
data1=cell2mat(data1);
disp(data1);
s=size(data1,1);
QV_1_ZH07_Z01_01 = [   0.985381744558545  -0.170360844946125  -0.000429350982799;
                       0.170360844946125   0.985381744558545   0.000312413931025;
                       0.000370009792980  -0.000382227096880   0.999999858314826;];
% QV_2_ZH09_Z01_01 = [   0.495205467375652   0.868774160271188  -0.001715657813041;
%                       -0.868767245771455   0.495208499972580   0.003405130790239;
%                        0.003808299222407  -0.000195476874979   0.999992728448308;];
faip=zeros(s,1);
thetap=zeros(s,1);
QV_2_ZH09_Z01_01=rotz(-60);
for i=1:s
    [faip(i),thetap(i)] = qv_kinematics(data1(i,1),data1(i,2), QV_2_ZH09_Z01_01);
end
% thetap_real=[];
% faip_real=[];
% 
% for i=1:size(faip)
%     err(i) = qv_err(faip(i),thetap(i),faip_real(i),thetap_real(i));
% end
% err=err';