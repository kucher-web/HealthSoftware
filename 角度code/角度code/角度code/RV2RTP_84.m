function [d]=RV2RTP_84(theta,phi,RV,Q)
%%
% 计算本体坐标系下的单位方向矢量,本体坐标系和天线坐标系一样，不重复写了
  v_body = [cosd(90-theta) * cosd(phi);
          cosd(90-theta) * sind(phi);
          sind(90-theta)];
  %% 本体到质心轨道
  A=[2*(Q(1)^2+Q(4)^2)-1 2*(Q(1)*Q(2)+Q(3)*Q(4)) 2*(Q(1)*Q(3)-Q(2)*Q(4))
    2*(Q(1)*Q(2)-Q(3)*Q(4)) 2*(Q(2)^2+Q(4)^2)-1 2*(Q(2)*Q(3)+Q(1)*Q(4))
    2*(Q(1)*Q(3)+Q(2)*Q(4)) 2*(Q(2)*Q(3)-Q(1)*Q(4)) 2*(Q(3)^2+Q(4)^2)-1];
  A1=inv(A);
  porb=A1*v_body;
  %% 轨道系到84系，修正速度值
R=[];
omg = 7.2921151467e-5;%用于修正地球自转转速
r = RV(1:3);
v = RV(4:6) + omg*[-r(2) r(1) 0];
R(3,:) = -r/norm(r);
R(2,:) = cross(v,r)/norm(cross(v,r));
R(1,:) = cross(R(2,:),R(3,:));
R1=inv(R);
d=R1*porb;%生成在84坐标系下的位置坐标矢量

end