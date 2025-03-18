class TMO:
    def __init__(self):
        self.TMO108 = None  # 短时外推时间秒
        self.TMO109 = None  # 短时外推时间微秒
        self.TMO110 = None  # 短时外推位置X
        self.TMO111 = None  # 短时外推位置Y
        self.TMO112 = None  # 短时外推位置Z
        self.TMO113 = None  # 短时外推速度X
        self.TMO114 = None  # 短时外推速度Y
        self.TMO115 = None  # 短时外推速度Z

    def update(self, data):
        # 更新轨道参数
        pass

    def calculate_position(self):
        # 计算位置
        pass

class TMK:
    def __init__(self):
        self.TMK505 = None  # TMK505_轨道系姿态角EulerX
        self.TMK506 = None  # TMK506_轨道系姿态角EulerY
        self.TMK507 = None  # TMK507_轨道系姿态角EulerZ
        self.TMK508 = None  # TMK508_轨道系角速度wbox
        self.TMK509 = None  # TMK509_轨道系角速度wboy
        self.TMK510 = None  # TMK510_轨道系角速度wboz

    def update(self, data):
        # 更新姿态参数
        pass

    def calculate_attitude(self):
        # 计算姿态
        pass

class TMZ:
    def __init__(self):
        self.TMZ148 = None  # TMZ148_馈电指向时间_秒
        self.TMZ150 = None  # TMZ150_馈电1俯仰角
        self.TMZ151 = None  # TMZ151_馈电1方位角
        self.TMZ152 = None  # TMZ152_馈电2俯仰角
        self.TMZ153 = None  # TMZ153_馈电2方位角
        self.TMZ067 = None  # TMZ067_QV1轴1（副面轴）转台绝对位置
        self.TMZ068 = None  # TMZ068_QV1轴2（主面轴）转台绝对位置

    def update(self, data):
        # 更新遥测参数
        pass

    def process_telemetry(self):
        # 处理遥测数据
        pass

class Satellite_angle:
    # 定义卫星对象，拥有上述三个对象以及时间属性
    def __init__(self):
        self.tmz = TMZ()
        self.tmk = TMK()
        self.tmo = TMO()
        self.time = None
        
    def update(self, data):
        # 更新卫星参数
        pass
        
    def calculate(self):
        # 计算卫星姿态
        pass
    
    def process(self):
        # 处理卫星数据
        pass

    def display(self):
        # 展示卫星数据
        for key in self.__dict__:
            # 展示TMO数据
            if key == 'tmo':
                print('TMO:',self.tmo.__dict__)
            # 展示TMK数据
            elif key == 'tmk':
                print('TMK:',self.tmk.__dict__)
            # 展示TMZ数据
            elif key == 'tmz':
                print('TMZ:',self.tmz.__dict__)
            # 展示时间数据
            elif key == 'time':
                print('Time:',self.time)
        pass