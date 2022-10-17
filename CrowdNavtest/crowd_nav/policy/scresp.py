from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
import numpy as np
from numpy.linalg import norm


class ScResp(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'ScResp'
        self.kinematics = None
        self.gamma = None
        self.with_om = None
        self.radius = 0.3
        self.d_comfortable = 0.2
        self.laststate = None
        self.lasttime = 0
        self.dmin = 10

        self.k_goal = 1.  # 目标吸引力系数
        self.k_multi_r_so = 0.3  # 多点情况静态目标排斥力系数
        self.k_multi_l_so = 0.6  # 多点情况静态目标切向力系数
        self.k_CL_hv = 1.0  # 普通相机区假定的机器人速度系数

        self.k_single_r_do = 0.3  # 单点情况动态目标排斥力系数
        self.k_single_l_do = 0.3  # 单点情况动态目标切向力系数
        self.k_multi_r_do = 0.6  # 单点情况动态目标排斥力系数
        self.k_multi_l_do = 0.3  # 单点情况动态目标切向力系数

        self.th_r_so = 0.2
        self.th_l_so = 0.2
        self.th_threat = 0
        self.th_r_do = 0.2
        self.th_l_do = 0.3
        self.th_single_do_exrange = 0.2

    def configure(self, config):
        self.gamma = 0.9
        self.with_om = False
        self.multiagent_training = False

    def predict(self, state, h_id=0):
        robot_state = state.self_state  # (0,-4)->(0,4)
        # humans = self.view_range(robot_state, state.human_states)
        humans = state.human_states
        static_humans, dynamic_humans = self.human_classify(humans)


        # 将首选速度设置为目标方向上的单位幅度（速度）向量.
        velocity = np.array((robot_state.gx - robot_state.px, robot_state.gy - robot_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        reaching_goal = norm(velocity) < 0.3

        self.lasttime += 0.25
        if self.lasttime == 9.25:
            aaa = 0

        # 动态障碍路线避让

        dof = self.dynamic_obstacle_force(robot_state, dynamic_humans)
        # dof = self.dynamic_obstacle_force(robot_state, humans)

        # 静态障碍分区，所有点按距离划分区域
        static_obstacles = self.static_subarea(static_humans)

        sof = self.static_obstacle_force(robot_state, static_humans, static_obstacles)

        gf = self.goal_force(robot_state)
        if norm(dof+1e-5) > 0.8 or norm(sof+1e-5) > 0.8:
            gf = 0.15*gf
        F = gf + 1.5*sof/norm(sof+1e-5) + 1.*dof/norm(dof+1e-5)
        normF = norm(F)
        pref_v = F / normF if normF > 1 else F

        if reaching_goal:
            action = ActionXY(0, 0)
            print("dmin = %f", self.dmin)
        else:
            action = ActionXY(pref_v[0], pref_v[1])
        self.last_state = state

        return action

    def static_subarea(self, humans):
        # 构建障碍对
        obs_pair = []
        dis_pair = []
        D = 4 * self.radius + 2 * self.d_comfortable + 0.1  # 0.1是裕量
        for i in range(len(humans) - 1):
            for j in range(len(humans) - i - 1):
                obs_pair.append([i, i + j + 1])
                dis_pair.append(
                    norm(np.array((humans[i].px - humans[i + j + 1].px, humans[i].py - humans[i + j + 1].py))))
        # obs=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]  # 0,1,4,5
        # dis=[2.,3.,1.,1.,4.,5.]
        obs_ = []
        for i in range(len(humans)):
            obs_.append([i])
        for i in range(len(obs_)):
            for j in range(len(obs_pair)):
                if obs_[i][0] in obs_pair[j]:
                    if dis_pair[j] < D:
                        obs_[i] += obs_pair[j]

        def inter(a, b):  # 判断交集是否存在
            return list(set(a) & set(b))

        for i in range(len(obs_)):
            j = 0
            if i >= len(obs_) - 1:
                break
            while 1:
                if i + j == len(obs_) - 1:
                    break
                elif inter(obs_[i], obs_[i + j + 1]):
                    obs_[i] += obs_[i + j + 1]
                    obs_.remove(obs_[i + j + 1])
                else:
                    j += 1

        obs_final = [list(set(i)) for i in obs_]

        return obs_final

    def static_obstacle_force(self, rs, hs, ob):
        sof = np.array((0., 0.))
        if len(ob) == 0:
            return sof
        for i in range(len(ob)):
            sof += self.multi_sof(rs, hs, ob[i])
        return sof

    def multi_sof(self, rs, humans, ob):
        p = np.array((rs.px, rs.py))
        pg = np.array((rs.gx, rs.gy))
        dis_pg = self.dis(p, pg)
        phs = []
        dis_ph = []
        hrg_ang = []
        for _ob in ob:
            phs += [np.array((humans[_ob].px, humans[_ob].py))]
            dis_ph += [self.dis(p, phs[-1])]
            hrg_ang += [self.In_angle(phs[-1], p, pg)]
        index_dis_ph = np.argsort(dis_ph)  # 距离从小到大下标

        D = 2 * self.radius
        pcenter = sum(phs)/len(phs)
        msof = np.array((0., 0.))
        i = -1  # 用于决定切向力角度

        if dis_ph[index_dis_ph[0]] < self.dmin:
            self.dmin = dis_ph[index_dis_ph[0]] - D

        r_center = self.dis(phs[index_dis_ph[0]], phs[index_dis_ph[-1]])/2 + self.radius
        detect_range = r_center + self.radius
        dis_cr = self.dis(p, pcenter)  # 中心点到机器人距离
        if dis_pg > dis_cr:
            if dis_cr <= detect_range:  # 中心进入范围
                dis_line = self.point_distance_line(pcenter, p, pg)  # 中心点到机目线的距离
                crg_ang = self.In_angle(pcenter, p, pg)
                if dis_line <= detect_range:  # 干扰路线
                    if crg_ang < 0.5 * np.pi or crg_ang > -0.5 * np.pi:  # 中心点在前方
                        if dis_cr < dis_pg:  # 中心点必须在机目中间
                            th_cD = r_center
                            if self.In_angle(pcenter, p, pg) > 0:
                                i = 1
                            vec_repu = (p - pcenter)/norm(p - pcenter)
                            vec_tang = self.rotate(vec_repu, i * 0.5 * np.pi)
                            # F_repu = np.abs(self.k_multi_r_so * (1 / (dis_cr - th_cD) ** 2 - 1 /
                            #                                      r_center ** 2)) * vec_repu  # 排斥力
                            # F_tang = np.abs(self.k_multi_l_so * (1 / (dis_cr - th_cD) ** 2 - 1 /
                            #                                      r_center ** 2)) * vec_tang  # 切向力
                            F_repu = np.abs(self.k_multi_r_so * (1 / dis_cr)) * vec_repu  # 排斥力
                            F_tang = np.abs(self.k_multi_l_so * (1 / dis_cr)) * vec_tang  # 切向力
                            msof += (F_repu + F_tang)/norm(F_repu + F_tang)

        detect_range = self.th_r_so+0.2
        if dis_ph[index_dis_ph[0]] - D <= detect_range:  # 单点进入范围
            for _ob in index_dis_ph:
                if dis_ph[index_dis_ph[_ob]] - D <= detect_range:
                    dis_line = self.point_distance_line(phs[index_dis_ph[_ob]], p, pg)  # 最近点到机目线的距离
                    if dis_line - D <= self.th_l_so:  # 干扰路线
                        if hrg_ang[index_dis_ph[_ob]] < 0.5 * np.pi or hrg_ang[index_dis_ph[_ob]] > -0.5 * np.pi:  # 最近点在前方
                            if dis_ph[index_dis_ph[_ob]] < dis_pg:  # 最近点必须在机目中间
                                pnear = phs[index_dis_ph[_ob]]
                                dis_cn = self.dis(pnear, pcenter)  # 中心点到最近点距离
                                dis_hr = self.dis(p, pnear)
                                D = dis_cn + 2*self.radius + self.d_comfortable
                                th_D = 2*self.radius + self.d_comfortable
                                if self.In_angle(pcenter, p, pg) > 0:
                                    i = 1
                                vec_repu = (p - pnear)/norm(p - pnear)
                                vec_tang = self.rotate(vec_repu, i * 0.5 * np.pi)
                                F_repu = np.abs(self.k_multi_r_so * (1 / (dis_hr - th_D) ** 2 - 1 /
                                                                     self.th_r_so ** 2)) * vec_repu  # 排斥力
                                F_tang = np.abs(self.k_multi_l_so * (1 / (dis_hr - th_D) ** 2 - 1 /
                                                                     self.th_l_so ** 2)) * vec_tang  # 切向力
                                msof += F_repu + F_tang

                                # 计算中心对机器人的力,有几个避开的障碍计算几遍
        return msof

    def goal_force(self, rs):
        p = np.array((rs.px, rs.py))
        pg = np.array((rs.gx, rs.gy))
        gf = self.k_goal * (pg - p) / self.dis(p, pg)
        return gf

    def dis(self, p, q):
        distance = norm(q - p)

        return distance

    def In_angle(self, A, O, B):  # 向量夹角，顺时针为正
        a = A - O
        b = B - O
        a /= norm(a+1e-4)
        b /= norm(b+1e-4)
        # 夹角cos值
        cos_ = np.dot(a, b) / (norm(a+1e-4) * norm(b+1e-4))
        # 夹角sin值
        sin_ = np.cross(a, b) / (norm(a+1e-4) * norm(b+1e-4))
        arctan2_ = np.arctan2(sin_, cos_)

        return arctan2_

    def vec_angle(self, A, B):  # A相对于B的向量夹角，顺时针为正
        a = A/norm(A+1e-4)
        b = B/norm(B+1e-4)
        # 夹角cos值
        cos_ = np.dot(a, b) / (norm(a+1e-4) * norm(b+1e-4))
        # 夹角sin值
        sin_ = np.cross(a, b) / (norm(a+1e-4) * norm(b+1e-4))
        arctan2_ = np.arctan2(sin_, cos_)

        return arctan2_

    def rotate(self, a, alpha):
        x_ = a[1] * np.sin(alpha) + a[0] * np.cos(alpha)
        y_ = a[1] * np.cos(alpha) - a[0] * np.sin(alpha)
        return np.array((x_, y_))

    def point_distance_line(self, point, line_point1, line_point2):  # 点到直线距离
        # 计算向量
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
        return distance

    def view_range(self, robot, humans):  # 机器人视角范围
        p = np.array((robot.px, robot.py))
        v = np.array((robot.vx, robot.vy))
        pg = np.array((robot.gx, robot.gy))
        if norm(v) > 0.001:
            for human in humans:
                # human.vx = human.velocity.vx
                # human.vy = human.velocity.vy
                hp = np.array((human.px, human.py))
                angle_hrg = self.In_angle(hp, p, pg)
                dis_hr = self.dis(hp, p)
                if np.abs(angle_hrg) <= np.pi/180*35 and dis_hr <= 4.5:
                    aa=1  # 无改变
                elif np.abs(angle_hrg) <= np.pi/180*135 and dis_hr <= 10:
                    norm_v = norm(np.array((human.vx, human.vy)))
                    if norm_v > 0.001:
                        human.vx = self.k_CL_hv * human.vx / norm_v
                        human.vy = self.k_CL_hv * human.vy / norm_v
                else:
                    # norm_v = norm(np.array((human.vx, human.vy)))
                    # if norm_v > 0.001:
                    #     human.vx = self.k_CL_hv * human.vx / norm_v
                    #     human.vy = self.k_CL_hv * human.vy / norm_v
                    human.vx = 0
                    human.vy = 0
                    # human.px = -6
                    # human.py = -6

        return humans

    def human_classify(self, humans):
        static_humans = []
        dynamic_humans = []
        for human in humans:
            # if norm([float(human.vx), float(human.vy)]) < 0.001:
            if np.abs(human.vx) < 0.1 and np.abs(human.vy) < 0.1:
                static_humans.append(human)
            else:
                dynamic_humans.append(human)
        return static_humans, dynamic_humans

    def dynamic_obstacle_force(self, robot, humans):
        dof = np.array((0., 0.))
        if len(humans) == 0:
            return dof
        dof += self.multi_dof(robot, humans)
        return dof

    def multi_dof(self, robot, humans):
        p = np.array((robot.px, robot.py))
        v = np.array((robot.vx, robot.vy))
        pg = np.array((robot.gx, robot.gy))
        is_collision = []  # 是否相撞
        ctime = []  # 产生碰撞的时间
        problem = []  # 0无事发生，1超车，2被超车，3对碰， 4交叉
        cps = []  # 碰撞时机器人位置
        cphs = []  # 碰撞时人类位置
        phs = []  # 现在人类位置
        vhs = []  # 现在人类速度
        dis_ph = []

        for human in humans:
            phs.append(np.array((human.px, human.py)))
            vhs.append(np.array((human.vx, human.vy)))
            pgv = (pg-p)/norm(pg-p)
            dis_ph.append(self.dis(p,phs[-1]))
            _is_collision, _ctime, _problem, _cp, _cph = self.predict_intersection(p, pgv, phs[-1], vhs[-1])
            is_collision.append(_is_collision)
            if _is_collision == False:
                ctime.append(None)
                problem.append(None)
                cps.append(None)
                cphs.append(None)
            else:
                ctime.append(_ctime)
                problem.append(_problem)
                cps.append(_cp)
                cphs.append(_cph)

        dire = -1  # 默认切向力方向为逆时针
        F_repu = np.array((0., 0.))  # 排斥力
        F_tang = np.array((0., 0.))  # 切向力
        F_repu_every = np.array((0., 0.))
        F_tang_every = np.array((0., 0.))
        mdof = np.array((0., 0.))

        dynamic_obstacles = self.dynamic_subarea(cphs, ctime, humans)

        for dynamic_obstacle in dynamic_obstacles:  # 根据预测碰撞位置的避障
            cphs_group = [cphs[dynamic_obstacle[0]]]
            cvhs_group = [vhs[dynamic_obstacle[0]]]
            dis_ph_group = [self.dis(cps[dynamic_obstacle[0]], cphs[dynamic_obstacle[0]])]
            t = ctime[dynamic_obstacle[0]]
            for do in dynamic_obstacle:
                if do != dynamic_obstacle[0]:
                    human_pxy = np.array((humans[do].px, humans[do].py))
                    human_vxy = np.array((humans[do].vx, humans[do].vy))
                    human_pxy_now = human_pxy + human_vxy*t

                    cphs_group += [human_pxy_now]
                    cvhs_group += [human_vxy]
                    dis_ph_group += [self.dis(cps[dynamic_obstacle[0]], cphs_group[-1])]
            index_dis_ph = np.argsort(dis_ph_group)  # 距离从小到大下标
            cpcenter = sum(cphs_group)/len(cphs_group)
            cvcenter = sum(cvhs_group)/len(cvhs_group)
            cang_vg = self.vec_angle(cvcenter, pg-cps[dynamic_obstacle[0]])  # 按集群方向分配
            cang_crg = self.In_angle(cpcenter,cps[dynamic_obstacle[0]],pg)
            if cang_vg > 0:
                dire = 1
            # if cang_crg > 0:
            #     dire = 1
            r_center = self.dis(cphs_group[index_dis_ph[0]], cphs_group[index_dis_ph[-1]])/2 + self.radius+0.2
            th_cD = r_center
            cdis_cr = self.dis(cps[dynamic_obstacle[0]], cpcenter)
            dis_cr = self.dis(p, cpcenter)
            vec_repu = (cps[dynamic_obstacle[0]] - cpcenter)/norm(cps[dynamic_obstacle[0]] - cpcenter)  # 按碰撞位置与中心距离计算方向
            vec_repu = (p - cpcenter)/norm(p - cpcenter)
            vec_tang = self.rotate(vec_repu, dire * 0.5 * np.pi)

            F_repu = np.abs(self.k_multi_r_do/dis_cr * (1 /np.abs(cdis_cr) - 1 /
                                                        r_center)) * vec_repu  # 排斥力按碰撞时计算不会过大
            F_tang = np.abs(self.k_multi_l_do/dis_cr * (1 / np.abs(cdis_cr) - 1 /
                                                        r_center)) * vec_tang  # 切向力
            cc1 = F_repu + F_tang
            if len(index_dis_ph) > 1:
                mdof += (F_repu + F_tang)/len(index_dis_ph)

            for i in index_dis_ph:  # 全计算，与排序无关
                dis_collision = norm(v)*ctime[dynamic_obstacle[0]]
                cdis = self.dis(cps[dynamic_obstacle[0]], cphs_group[i]) - 2*self.radius
                dis_pc = self.dis(p, cphs_group[i])
                if cdis+1e-4 > 0:
                    vec_repu = (cps[dynamic_obstacle[0]] - cphs_group[i])/norm(cps[dynamic_obstacle[0]] - cphs_group[i])
                    vec_repu = (p - cphs_group[i])/norm(p - cphs_group[i])
                    vec_tang = self.rotate(vec_repu, dire * 0.5 * np.pi)
                    F_repu_every = self.k_multi_r_do / dis_pc * (1 / cdis) * vec_repu
                    F_tang_every = self.k_multi_l_do / dis_pc * (1 / cdis) * vec_tang
                    cc2 = F_repu_every + F_tang_every

                mdof += (F_repu_every + F_tang_every)/len(index_dis_ph)

        _index_dis_ph = 0
        mdsof = np.array((0., 0.))
        for i in range(len(humans)):
            if dis_ph[i] < 2*self.radius+self.d_comfortable+0.2:
                ang_hrg = self.In_angle(phs[i], p, pg)
                _index_dis_ph += 1
                ang_phv = self.In_angle(p, phs[i], phs[i]+vhs[i])
                dire = -1  # 默认切向力方向为逆时针
                if ang_phv > 0:
                    dire = 1
                vec_repu = (p - phs[i])/norm(p - phs[i])
                vec_tang = self.rotate(vec_repu, dire * 0.5 * np.pi)
                th_D = 2*self.radius+self.d_comfortable+0.1
                F_repu = np.abs(self.k_multi_l_do * (1 / (dis_ph[i] - th_D) ** 2 - 1 /
                                       0.3 ** 2)) * vec_repu  # 排斥力
                F_tang = np.abs(self.k_multi_l_do * (1 / (dis_ph[i] - th_D) ** 2 - 1 /
                                       0.3 ** 2)) * vec_tang  # 切向力
                mdsof += F_repu + F_tang
        if _index_dis_ph > 0:
            mdof += 0.6*mdsof/_index_dis_ph

        return mdof

    def predict_intersection(self, p, v, ph, vh):
        problem = 0  # 0无事发生，1超车，2被超车，3对碰， 4交叉
        is_collision = False  # 是否相撞
        ctime = 0  # 产生碰撞的时间
        # p = np.array((0., 0.))
        # v = np.array((1., 0.))
        # ph = np.array((1., 1.))
        # vh = np.array((0., -1.))
        _A = p
        _B = ph
        _C = np.array((0., 0.))
        angle_vector = self.vec_angle(v, vh)

        collision_time = self.min_dis(p, ph, v, vh, 2*self.radius+self.d_comfortable)
        if not collision_time is None:  # 碰撞
            #     is_collision = False
            # else:  # 存在碰撞
            is_collision = True
            ctime = collision_time[0]  # 有效的碰撞时间为第一个值
            _A = p+v*ctime
            _B = _A+v
            _C = ph+vh*ctime
            _D = _C+vh
            _angle_ACD = self.In_angle(_A, _C, _D)

            if np.abs(angle_vector) <= 0.25*np.pi:  # 方向均保持向前
                if np.abs(_angle_ACD) >= 0.5*np.pi:
                    problem = 1  # 超车
                else:
                    problem = 2  # 被超车
            elif np.abs(angle_vector) >= 0.75*np.pi:  # 方向均保持相对
                problem = 3  # 对撞
            else:
                problem = 4  # 交叉

        return is_collision, ctime, problem, _A, _C

    def quadratic_equation(self, a, b, c):  # a*x^2 + b*x + c = 0
        if a != 0:
            delta = b**2-4*a*c
            if delta < 0:
                # print("无根")
                return None
            elif delta == 0:
                s = -b/(2*a)
                # print("唯一的根x=",s)
                if s <= 0:
                    return None
                return [s, s]
            else:
                root = np.sqrt(delta)
                x1 = (-b-root)/(2*a)
                x2 = (-b+root)/(2*a)
                if x1 <= 0:
                    return None
                return [x1, x2]

    def min_dis(self, A, B, vA, vB, d):  # 求两直线上两动点最小间距
        # A=[A[0]+vA[0]*t, A[1]+vA[1]*t]
        # B=[B[0]+vB[0]*t, B[1]+vB[1]*t]
        # d^2 = (A[0]+vA[0]*t-B[0]-vB[0]*t)^2 + (A[1]+vA[1]*t-B[1]-vB[1]*t)^2
        #     = ((vA[0]-vB[0])*t+A[0]-B[0])^2 + ((vA[1]-vB[1])*t+A[1]-B[1])^2
        #     = (a1*t+b1)^2 + (a2*t+b2)^2
        #     = (a1^2+a2^2)*t^2 + 2*(a1*b1+a2*b2)*t + (b1^2+b2^2)
        # A=[2,0]
        # B=[0,0]
        # vA=[-1,0]
        # vB=[1,0]
        # d = 0.0  # 两动点间距
        a1 = vA[0]-vB[0]
        a2 = vA[1]-vB[1]
        b1 = A[0]-B[0]
        b2 = A[1]-B[1]
        a = a1*a1+a2*a2
        b = 2*(a1*b1+a2*b2)
        c = b1*b1+b2*b2-d*d
        # x,y =self.quadratic_equation_pole(a,b,c)
        motion_time = self.quadratic_equation(a,b,c)  # s[0] 是碰撞点，None则不碰撞
        return motion_time

    def dynamic_subarea(self, humans_p, time, humans):
        # 构建障碍对, 碰撞时同一时间离得近的组为集团
        obs_pair = []
        D = 4 * self.radius + 2 * self.d_comfortable + 0.1  # 0.1是裕量
        for i in range(len(humans_p)):
            if not humans_p[i] is None:
                _obs_pair = [i]
                for j in range(len(humans_p)):
                    if j != i:
                        human_pxy = np.array((humans[j].px, humans[j].py))
                        human_vxy = np.array((humans[j].vx, humans[j].vy))
                        human_pxy_now = human_pxy + human_vxy*time[i]
                        _dis = norm(humans_p[i] - human_pxy_now)
                        if _dis < D:
                            _obs_pair.append(j)
                obs_pair.append(_obs_pair)

        return obs_pair