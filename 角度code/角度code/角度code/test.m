    %%
    format longG
    % 输入用户位置，WGS84
    X=313437;
    Y=4770653;
    Z=4208987;%米为单位  新疆
    % X=-2839327;
    % Y=4679324;
    % Z=3263596;%米为单位 上海
    % X=-2973445;
    % Y=3072904;
    % Z=4716633;%米为单位  新疆
    USER_POS =[X Y Z];
    %%
    % 输入间隔64s的卫星位置，WGS84坐标系，单位米，指令TMO110~TMO115
    RV1=[-67623.57813	4548867.5	5540375.5	38.06939697	5776.424316	-4717.989746];
    RV2 =[-63341.4375	4908262	5226393.5	95.10617065	5450.527832	-5090.785645];
    RV3=[-55522.63281	5246013	4889252.5	148.4838715	5100.333496	-5441.143066];
    RV4=[-44421.28125	5560615.5	4530453	197.7364655	4727.712891	-5767.272949];
    RV5=[-30302.62305	5850656.5	4151589	242.703537	4333.28125	-6067.784668];
    RV6=[-13462.03516	6114912.5	3754381.5	282.7574463	3919.611816	-6341.501953];
    %% 
    % 做轨道插值
    k1=[RV2-RV1]/(64*1000);%线性插值每毫秒的轨道变化率
    k2=[RV3-RV2]/(64*1000);
    k3=[RV4-RV3]/(64*1000);
    k4=[RV5-RV4]/(64*1000);
    k5=[RV6-RV5]/(64*1000);
    RV1_=RV1-k1*2750;%星上时间，TMZ148整秒值比星上时少一秒，所以在星上时上少了1750ms
    RV2_=RV2-k1*2750;
    RV3_=RV3-k2*2750;
    RV4_=RV4-k3*2750;
    RV5_=RV5-k4*2750;
    RV6_=RV6-k5*2750;
    data1=[RV1_;RV2_;RV3_;RV4_;RV5_;RV6_];
    disp(data1);
    %%
    % 输入卫星姿态角 TMK505~TMK507
    alphax1=0.00083238; alphay1=-0.01101539;alphaz1=-0.00259131;
    alphax2=0.00175312; alphay2=-0.00096647;alphaz2=0.00285641;
    alphax3=0.00076279;alphay3=0.00370367;alphaz3=-0.00152585;
    alphax4=-0.00037315;alphay4=-0.0054522;alphaz4=-0.00001243;
    alphax5=0.00154832;alphay5=-0.01022821;alphaz5=0.00115596;
    alphax6=-0.00128882;alphay6=-0.00089305;alphaz6=-0.00061824;
    ALPHAX=[alphax1; alphax2;alphax3;alphax4;alphax5;alphax6]*pi/180;
    ALPHAY=[alphay1; alphay2;alphay3;alphay4;alphay5;alphay6]*pi/180;
    ALPHAZ=[alphaz1; alphaz2;alphaz3;alphaz4;alphaz5;alphaz6]*pi/180;
    %根据姿态角算四元数,6代表这一弧段需要计算6个点的时间
    Q1=zeros(6,4);
    Q2=zeros(6,4);
    Q3=zeros(6,4);
    Q1_STEP=zeros(6,4);
    Q2_STEP=zeros(6,4);
    for i=1:6
        Q1(i,:)=[sin(ALPHAX(i)/2) 0 0 cos(ALPHAX(i)/2)];
        Q2(i,:)=[0 sin(ALPHAY(i)/2) 0 cos(ALPHAY(i)/2)];
        Q3(i,:)=[0 0 sin(ALPHAZ(i)/2) cos(ALPHAZ(i)/2)];
        [Q1_STEP(i,:)]=siyuanshuchengfa(Q3(i,:),Q1(i,:));
        [Q2_STEP(i,:)]=siyuanshuchengfa(Q1_STEP(i,:),Q2(i,:));
    end
    disp(Q2_STEP);
    %已算出6个时刻的姿态四元数Q2_STEP
    %根据时间外推到整秒的四元数,首先输入角速度 TMK505~TMK510
    W_1=[-0.0007902	0.00033979	0.0002078];
    W_2=[-0.00164376	0.00156558	0.0001073];
    W_3=[-0.00053028	0.00036893	-0.00014961];
    W_4=[-0.00019678	0.0010008	-0.00027919];
    W_5=[-0.00209627	0.00168203	0.00033462];
    W_6=[-0.00314025	0.00112048	0.00085791
    ];
    W=zeros(6,3);
    Q=zeros(6,4);
    W=[W_1;W_2;W_3;W_4;W_5;W_6];
    for i=1:6
        Q(i,:)=OutExtra(Q2_STEP(i,:),W(i,:),-2.85);
    end
    %disp(Q);
    %已经外推到6个时刻的整秒值的四元数（轨道系）
    %输入安装矩阵
    AZJZ=1;
    %% 84到轨道系,切记此部分不能直接用J2000转轨道系的公式，需要修正速度值
    RV=ones(1,6);
    Qe=ones(1,4);
    theta=zeros(6,1);
    phi=zeros(6,1);
    rang=zeros(6,1);
    d=zeros(6,1);
    Bpos=zeros(6,3);
    Porb11=[];
    Porb12=[];

    for i=1:1:6
        RV =data1(i,:);
        Qe=Q(i,:);
        %84坐标系转轨道系，需利用每个时刻的卫星的速度值作修正
        R=[];
        omg = 7.2921151467e-5;%用于修正地球自转转速
        r = RV(1:3);
        v= RV(4:6) + omg*[-r(2) r(1) 0];
        R(3,:) = -r/norm(r);%Z轴指向卫星的位置向量的相反方向。因此，我们取卫星的位置向量r，并将其标准化（除以它的模长）以得到Z轴方向的单位向量。
        R(2,:) = cross(v,r)/norm(cross(v,r));%叉积，Y轴垂直于卫星的位置向量r和速度向量v。我们可以通过计算r和v的叉积来得到一个垂直于它们的向量，然后将其标准化以得到Y轴方向的单位向量。
        R(1,:) = cross(R(2,:),R(3,:));%X轴垂直于Y轴和Z轴。我们可以通过计算Y轴和Z轴的叉积来得到X轴方向的单位向量。
        % 生成用户在卫星轨道坐标系的坐标
        Porb1=USER_POS'-RV(1:3)';
        Porb11=[Porb11;Porb1];
        Porb1_=(USER_POS'-RV(1:3)')/norm(Porb1);
        Porb12=[Porb12;Porb1_];
        % Porb12=reshape(Porb12,6,3);
        Porb=R*(USER_POS'-RV(1:3)');%指向向量
        %% 第三步，质心轨道到本体
    % 根据四元素计算姿态矩阵
    % 第二种姿态四元素表示法 q1~q4
        A=[2*(Qe(1)^2+Qe(4)^2)-1 2*(Qe(1)*Qe(2)+Qe(3)*Qe(4)) 2*(Qe(1)*Qe(3)-Qe(2)*Qe(4))
        2*(Qe(1)*Qe(2)-Qe(3)*Qe(4)) 2*(Qe(2)^2+Qe(4)^2)-1 2*(Qe(2)*Qe(3)+Qe(1)*Qe(4))
        2*(Qe(1)*Qe(3)+Qe(2)*Qe(4)) 2*(Qe(2)*Qe(3)-Qe(1)*Qe(4)) 2*(Qe(3)^2+Qe(4)^2)-1];
        Psat=A*Porb;
        Bpos(i,:)=(AZJZ*Psat)';
        %% 计算方位角俯仰角和距离
        %计算用户位置在天线坐标系下的方位角，俯仰角
        theta(i) = atand(sqrt(Bpos(i,1)^2+Bpos(i,2)^2)/Bpos(i,3));%%%俯仰角，单位度数
        phi(i)= atand((Bpos(i,2)/Bpos(i,1)));%%%方位角，单位度数
        rang(i)=sqrt(Bpos(i,1)^2+Bpos(i,2)^2+Bpos(i,3)^2);

        if(theta(i)<0 )
            theta(i)=180+theta(i);
        end
    
        if(Bpos(i,1)<0 && Bpos(i,2)>0)  %%二象
            phi(i)=180+phi(i);
        end
        if(Bpos(i,1)<0 && Bpos(i,2)<0)  %%三象
            phi(i)=phi(i)+180;
        end
        if(Bpos(i,1)>0 && Bpos(i,2)<0)  %%四象
            phi(i)=360+phi(i);
        end 
    end
    disp(theta);
    disp(phi);
    %Porb12=reshape(Porb12,6,3);