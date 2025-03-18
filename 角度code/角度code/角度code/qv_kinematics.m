function [faip,thetap] = qv_kinematics(thetay,thetax,R_mount)
%输入 thetay-副面轴角deg；thetax-主面轴角deg；R_mount-安装矩阵；输出faip-方位角deg；thetap-俯仰角deg；
gama = 29; %deg,固定角度，与天线设计有关；
qv = R_mount*roty(gama)*roty(thetay)*rotx(thetax)*roty(gama)'*[0 0 1]';
%%
%x y z为天线指向单位矢量
x = qv(1);
y = qv(2);
z = qv(3);
%单位矢量化
x = x/norm([x,y,z]);
y = y/norm([x,y,z]);
z = z/norm([x,y,z]);

thetap = acosd(z); %俯仰角
if y>=0
    faip = atan2(y,x)*180/pi; %方位角
% elseif x<=0 && y>=0
%     faip = atan2(y,x)*180/pi; %方位角
elseif y<0 
    faip = 360 + atan2(y,x)*180/pi; %方位角
% elseif x>0 && y<0
%     faip = 360 + atan2(y,x)*180/pi %方位角
end

% if x >= 0 && y>=0
%     faip = atan2(y,x)*180/pi; %方位角
% elseif x<=0 && y>=0
%     faip = atan2(y,x)*180/pi; %方位角
% elseif y<=0 
%     faip = 360 + atan2(y,x)*180/pi; %方位角
% % elseif x>0 && y<0
% %     faip = 360 + atan2(y,x)*180/pi %方位角
% end
% rotz(faip(k,1))*roty(theta(k,1))*[0 0 1]';
% disp(['天线',num2str(QV_ID),'的方位角是',num2str(faip(k,1)),'deg，俯仰角是',num2str(theta(k,1)),'deg']);
end
% faip %方位角
% theta %俯仰角

% subplot(2,1,1);
% plot(thetax,faip)
% title('方位角曲线')
% legend('方位角位置曲线')
% xlabel('主面轴角度（deg）')
% ylabel('角度（deg）')
% grid on
% subplot(2,1,2); 
% plot(thetax,theta)
% title('俯仰角曲线')
% legend('俯仰角位置曲线')
% xlabel('主面轴角度（deg）')
% ylabel('角度（deg）')
% grid on
% filename = 'QV_data.xlsx';
% A = {'主面轴角/deg','方位角/deg','俯仰角/deg';thetax, faip, theta};
% xlswrite(filename,A)















