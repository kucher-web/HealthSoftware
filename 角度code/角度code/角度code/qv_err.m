function err = qv_err(faip_lilun,thetap_lilun,faip_real,thetap_real)
%����faip-��λ�ǣ�thetap-�����ǣ�real-��ʾʵ��ֵ��lilun-��ʾ����ֵ����λdeg��
%���err-ָ��ƫ��Ƕȣ�ʸ���нǣ���λdeg
temp_lilun = rotz(faip_lilun)*roty(thetap_lilun)*[0;0;1];
temp_real = rotz(faip_real)*roty(thetap_real)*[0;0;1];
err = acosd(dot(temp_lilun,temp_real)/(norm(temp_lilun)*norm(temp_real)));

end

