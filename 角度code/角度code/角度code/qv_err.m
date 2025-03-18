function err = qv_err(faip_lilun,thetap_lilun,faip_real,thetap_real)
%输入faip-方位角，thetap-俯仰角，real-表示实测值；lilun-表示理论值；单位deg；
%输出err-指向偏差角度，矢量夹角；单位deg
temp_lilun = rotz(faip_lilun)*roty(thetap_lilun)*[0;0;1];
temp_real = rotz(faip_real)*roty(thetap_real)*[0;0;1];
err = acosd(dot(temp_lilun,temp_real)/(norm(temp_lilun)*norm(temp_real)));

end

