#!/usr/bin/env python3
import math
import threading
import numpy as np
import rospy

from kortex_driver.msg import BaseCyclic_Feedback, JointSpeed
from kortex_driver.srv import SendJointSpeedsCommand, SendJointSpeedsCommandRequest
from sensor_msgs.msg import ChannelFloat32, JointState
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import TransformStamped, Vector3Stamped
from std_msgs.msg import Float64, Int32MultiArray
import tf2_ros
import tf.transformations as tr
from scipy.spatial.transform import Rotation


EE_R_H_OFFSET = np.array([[ 0.0, 1.0, 0.0],
                          [1.0,0.0, 0.0],
                           [0.0,0.0,-1.0]])  # Offset from Geomagic Touch to Kinova Gen3 end-effector
def dh_matrix(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0,              np.sin(alpha),                 np.cos(alpha),                d],
        [0,              0,                             0,                            1]
    ])


def compute_fk_and_jacobian(q, dh_params, base_transform=np.eye(4)):
    num_joints = len(q)
    T = base_transform.copy()

    o = [T[0:3, 3].copy()]
    z = [T[0:3, 2].copy()]

    for i in range(num_joints):
        theta_offset, d, a, alpha = dh_params[i]
        theta_total = q[i] + theta_offset

        A_i = dh_matrix(theta_total, d, a, alpha)
        T = T @ A_i

        o.append(T[0:3, 3].copy())
        z.append(T[0:3, 2].copy())

    o_n = o[-1]

    J = np.zeros((6, num_joints))
    for i in range(num_joints):
        J[0:3, i] = np.cross(z[i], (o_n - o[i]))
        J[3:6, i] = z[i]

    return T, J

def rot_to_euler_parameters(R):
    # Markley Algorithm (2008)
    trR = np.trace(R)
    
    candidates = np.array([R[0,0], R[1,1], R[2,2], trR])
    idx = np.argmax(candidates)
    
    if idx == 0:  
        x = np.array([
            1.0 + R[0, 0] - R[1, 1] - R[2, 2],
            R[0, 1] + R[1, 0],
            R[0, 2] + R[2, 0],
            R[1, 2] - R[2, 1]
        ])
    elif idx == 1:  
        x = np.array([
            R[1, 0] + R[0, 1],
            1.0 + R[1, 1] - R[2, 2] - R[0, 0],
            R[1, 2] + R[2, 1],
            R[2, 0] - R[0, 2]
        ])
    elif idx == 2: 
        x = np.array([
            R[2, 0] + R[0, 2],
            R[2, 1] + R[1, 2],
            1.0 + R[2, 2] - R[0, 0] - R[1, 1],
            R[0, 1] - R[1, 0]
        ])
    else:  
        x = np.array([
            R[1, 2] - R[2, 1],
            R[2, 0] - R[0, 2],
            R[0, 1] - R[1, 0],
            1.0 + trR
        ])
        
    norm_x = np.linalg.norm(x)
    q = x/norm_x
    return q[3], q[0:3]  # eta, eps

def compute_orientation_error_euler(R_d, R_e):
    """
    Calcula o erro de orientação baseado estritamente em Parâmetros de Euler
    (Caccavale et al., 1999 - Eq. 11 e Eq. 27).
    Livre de singularidades e contínuo em todo o espaço SO(3).
    """
    # 1. Rotação relativa no referencial do efetuador: R_de = R_e^T @ R_d (Eq. 11)
    R_de = R_e.T @ R_d

    # 2. Extração pronta e estável do quatérnio via SciPy
    if hasattr(Rotation, "from_matrix"):
        rot = Rotation.from_matrix(R_de)
    else:
        rot = Rotation.from_dcm(R_de)

    # SciPy retorna no formato [x, y, z, w] -> eps = q[0:3], eta = q[3]
    q = rot.as_quat()
    eps_de_e = q[0:3]
    eta_de = q[3]

    # 3. Escolha da representação no menor arco de rotação (theta in [-pi, pi]):
    # Conforme Eq. (6) de Caccavale (1999), eta >= 0 garante unicidade e estabilidade
    if eta_de < 0.0:
        eps_de_e = -eps_de_e

    # 4. Transforma o vetor de erro para o referencial da base (Eq. 27):
    # eps_de = R_e @ eps_de_e
    eps_de = R_e @ eps_de_e

    return eps_de

class TeleopNode(object):
    def __init__(self):
        self.arm_joint_state_topic = rospy.get_param(
            "~arm_joint_state_topic", "/my_gen3/joint_states"
        )
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.err_rot_pub = rospy.Publisher("/teleop/orientation_error", Vector3Stamped, queue_size=1)
        self.err_norm_pub = rospy.Publisher("/teleop/orientation_error_norm", Float64, queue_size=1)
        
        self.haptic_joint_topic = rospy.get_param("~haptic_joint_topic", "/geomagic/joint_states")
        self.button_topic = rospy.get_param("~button_topic", "/geomagic/button")
        self.wrench_topic = rospy.get_param("~wrench_topic", "/my_gen3/base_feedback")
        self.force_topic = rospy.get_param("~force_topic", "/arm/force_feedback")
        self.gain = rospy.get_param("~position_gain", 0.9)
        self.orientation_gain = rospy.get_param("~orientation_gain", 0.4)
        self.workspace_scale = rospy.get_param("~workspace_scale", 3.0)
        self.max_deg_s = rospy.get_param("~max_joint_speed_deg_s", 35.0)
        self.damping = rospy.get_param("~damping_lambda", 0.02)
        self.send_zero_when_disabled = rospy.get_param("~send_zero_when_disabled", True)
        self.robot_ns = rospy.get_param("~robot_service_ns", "my_gen3")
        
        self.kinova_base_frame = rospy.get_param("~kinova_base_frame", "base_link")
        self.geomagic_base_frame = rospy.get_param("~geomagic_base_frame", "base")
        
        self.enabled = False
        self._q_arm = None
        self._hap0 = None
        self._ee0 = None
        self._R_hap0 = None
        self._R_ee0 = None

        self.force_scale = rospy.get_param("~force_scale", 1.0)
        self.force_deadband = rospy.get_param("~force_deadband", 0.5)
        self.max_feedback_force = rospy.get_param("~max_feedback_force", 5.0)
        self.force_filter_alpha = rospy.get_param("~force_filter_alpha", 0.2)
        self._f_filtered = np.zeros(3)
        self._wrench_raw = np.zeros(3)
        self._lock = threading.Lock()

        # DH Parameters
        self.hap_dh_params = [
            (0.0,       0.125,   0.0,   np.pi / 2),
            (0.0,       0.0,     0.14,  0.0),
            (0.0,       0.0,     0.0,   np.pi / 2),
            (0.0,       0.1325,  0.0,  -np.pi / 2),
            (np.pi / 2, 0.0,     0.0,  -np.pi / 2),
            (0.0,       0.0,    0.0,   0.0)  ## Frame stylus
        ]
        self.hap_joint_names = ['waist', 'shoulder', 'elbow', 'yaw', 'pitch', 'roll']

        self.arm_base_transform = dh_matrix(0.0, 0.0, 0.0, np.pi)
        self.arm_dh_params = [
            (0.0,       -(0.1564 + 0.1284), 0.0, np.pi / 2),
            (np.pi,     -(0.0054 + 0.0064), 0.0, np.pi / 2),
            (np.pi,     -(0.2104 + 0.2104), 0.0, np.pi / 2),
            (np.pi,     -(0.0064 + 0.0064), 0.0, np.pi / 2),
            (np.pi,     -(0.2084 + 0.1059), 0.0, np.pi / 2),
            (np.pi,      0.0,               0.0, np.pi / 2),
            (np.pi,     -(0.1059 + 0.0615), 0.0, np.pi)
        ]
        self.arm_joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"
        ]

        n = len(self.arm_dh_params)
        self._last_qdot = np.zeros(n)

        rospy.loginfo("Arm joints (%d): %s", n, self.arm_joint_names)
        service = "/" + self.robot_ns.lstrip("/") + "/base/send_joint_speeds_command"
        rospy.loginfo("Waiting for %s", service)
        rospy.wait_for_service(service, timeout=120.0)
        self._set_speed = rospy.ServiceProxy(service, SendJointSpeedsCommand)
        
        self.orientation_engaged = False
        self.orientation_tolerance_rad = np.deg2rad(15.0) # 15 graus de tolerância
        self.R_map = np.eye(3)

        rospy.Subscriber(self.arm_joint_state_topic, JointState, self._on_arm_state, queue_size=1)
        rospy.Subscriber(self.haptic_joint_topic, JointState, self._on_haptic_state, queue_size=1)
        rospy.Subscriber(self.button_topic, Int32MultiArray, self._on_button, queue_size=1)
        rospy.Subscriber(self.wrench_topic, BaseCyclic_Feedback, self._on_wrench, queue_size=1)
        self._force_pub = rospy.Publisher(self.force_topic, ChannelFloat32, queue_size=1)

        hz = rospy.get_param("~control_rate_hz", 40.0)
        rospy.Timer(rospy.Duration(1.0 / hz), self._on_control)

    def _base_velocity(self, v_des,omega_des):
        """Map desired EE linear velocity in base frame to joint velocities (damped LS)."""
        T, J = compute_fk_and_jacobian(self._q_arm, self.arm_dh_params, base_transform=self.arm_base_transform)
        twist_des = np.concatenate((v_des, omega_des))
        jjt = J @ J.T + self.damping * np.eye(6)
        qdot = J.T @ np.linalg.solve(jjt, twist_des)
        return qdot

    def _on_button(self, msg):
        if not msg.data or len(msg.data) < 2:
            return
        if msg.data[0] == 1 and msg.data[1] == 0:
            if not self.enabled:
                self.enabled = True
                self.orientation_engaged = False  # Exige checagem de alinhamento
                self._hap0 = None
                self._ee0 = None
                rospy.loginfo("Teleop ativado. Aguardando alinhamento da caneta com o robô...")
                
        if msg.data[1] == 1 and msg.data[0] == 0:
            self.enabled = False
            self.orientation_engaged = False
            rospy.loginfo("Teleop desativado.")
            if self.send_zero_when_disabled:
                self._send_speeds(np.zeros(len(self.arm_dh_params)))

    def _on_arm_state(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        self._q_arm = np.array([name_to_pos.get(name, 0.0) for name in self.arm_joint_names])
        T_ee_base, _ = compute_fk_and_jacobian(self._q_arm, self.arm_dh_params, base_transform=self.arm_base_transform)
        self._broadcast_tf(T_ee_base, self.kinova_base_frame, "calculated_kinova_ee")
        

    def _on_wrench(self, msg):
        w = np.array([
            msg.base.tool_external_wrench_force_x,
            msg.base.tool_external_wrench_force_y,
            msg.base.tool_external_wrench_force_z,
        ])
        with self._lock:
            self._wrench_raw = w
            
            
    def _broadcast_tf(self, T, parent_frame, child_frame):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        t.transform.translation.x = float(T[0, 3])
        t.transform.translation.y = float(T[1, 3])
        t.transform.translation.z = float(T[2, 3])

        quat = tr.quaternion_from_matrix(T)
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])

        self.tf_broadcaster.sendTransform(t)

    def _on_haptic_state(self, msg):
        if self._q_arm is None or not self.enabled:
            self._last_qdot = np.zeros(len(self.arm_dh_params))
            return

        # 1. Cinemática Direta do Geomagic (p, R_h)
        name_to_pos = dict(zip(msg.name, msg.position))
        q_h = np.array([name_to_pos.get(name, 0.0) for name in self.hap_joint_names])
        T_h, _ = compute_fk_and_jacobian(q_h, self.hap_dh_params)
        p = T_h[0:3, 3]
        R_h = T_h[0:3, 0:3]

        # 2. Cinemática Direta do Kinova Gen3 (p_cur, R_cur)
        Tc, _ = compute_fk_and_jacobian(
            self._q_arm, self.arm_dh_params, base_transform=self.arm_base_transform
        )
        p_cur = Tc[0:3, 3]
        R_cur = Tc[0:3, 0:3]

        # 3. Âncora de posição para translação relativa com escala
        if self._hap0 is None or self._ee0 is None:
            self._hap0 = p.copy()
            self._ee0 = p_cur.copy()
            return

        # 4. Translação Desejada (Relativa com embreagem)
        delta_pos = p - self._hap0
        p_des = self._ee0 + self.workspace_scale * delta_pos
        err_pos = p_des - p_cur

        # 5. Orientação Desejada (Absoluta com base alinhada)
        R_des = self.R_map @ R_h @ EE_R_H_OFFSET   
        
        T_des = np.eye(4)
        T_des[0:3, 0:3] = R_des
        T_des[0:3, 3] = p_des
        self._broadcast_tf(T_des, self.kinova_base_frame, "target_ee_des") 
        
        R_error = R_cur.T @ R_des

        cos_theta = (np.trace(R_error) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angle_error = np.arccos(cos_theta)

        print("R_des:\n", np.round(R_des, 3))
        print("R_cur:\n", np.round(R_cur, 3))
        print("R_error:\n", np.round(R_error, 3))
        print("Erro angular [graus]:", np.rad2deg(angle_error))

        # 6. Cálculo do erro angular
        err_R = compute_orientation_error_euler(R_des, R_cur)
        # print ("Erro de orientação (rad):", err_R)
        angle_error_norm = np.linalg.norm(err_R)
        # print ("Norma do erro de orientação (rad):", angle_error_norm)
        
        msg_vec = Vector3Stamped()
        msg_vec.header.stamp = rospy.Time.now()
        msg_vec.header.frame_id = self.kinova_base_frame
        msg_vec.vector.x = float(err_R[0])
        msg_vec.vector.y = float(err_R[1])
        msg_vec.vector.z = float(err_R[2])
        self.err_rot_pub.publish(msg_vec)

        self.err_norm_pub.publish(Float64(data=angle_error_norm))


        # 7. Lógica da Guarda de Reengate (Orientation Match)
        if not self.orientation_engaged:
            if angle_error_norm < self.orientation_tolerance_rad:
                self.orientation_engaged = True
                rospy.loginfo(">>> Orientação alinhada! Teleoperação ativa.")
            else:
                # Enquanto a mão não estiver aproximadamente alinhada com a ponta, congela os motores
                rospy.logwarn_throttle(
                    1.0, 
                    "Desalinhamento: %.1f deg. Aproxime a atitude da caneta da ferramenta (Limite: %.1f deg)", 
                    np.rad2deg(angle_error_norm), 
                    np.rad2deg(self.orientation_tolerance_rad)
                )
                self._last_qdot = np.zeros(len(self.arm_dh_params))
                return

        # 8. Malha de Velocidade Cartesiana DLS
        v = self.gain * err_pos
        omega = self.orientation_gain * err_R

        self._last_qdot = self._base_velocity(v, omega)
        
    def _publish_force_feedback(self, wrench):
        f = wrench * self.force_scale
        for i in range(3):
            if abs(f[i]) < self.force_deadband:
                f[i] = 0.0
        f = np.clip(f, -self.max_feedback_force, self.max_feedback_force)
        self._f_filtered = (
            (1.0 - self.force_filter_alpha) * self._f_filtered
            + self.force_filter_alpha * f
        )
        msg = ChannelFloat32()
        msg.name = "fx_fy_fz"
        msg.values = [float(self._f_filtered[0]), float(self._f_filtered[1]), float(self._f_filtered[2])]
        self._force_pub.publish(msg)

    def _on_control(self, _evt):
        if not self.enabled or self._q_arm is None:
            return
        with self._lock:
            wrench = self._wrench_raw.copy()
        self._send_speeds(self._last_qdot)
        self._publish_force_feedback(wrench)

    def _send_speeds(self, qdot_rad_s):
        max_r = self.max_deg_s * math.pi / 180.0
        q = np.clip(qdot_rad_s, -max_r, max_r)
        req = SendJointSpeedsCommandRequest()
        for i, w in enumerate(q):
            req.input.joint_speeds.append(JointSpeed(i, w * 180.0 / math.pi, 0))
        try:
            self._set_speed(req)
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(2.0, "send_joint_speeds: %s", e)

    def run(self):
        rospy.loginfo("geomagic_kinova_cartesian_teleop running.")
        rospy.spin()


def main():
    rospy.init_node("geomagic_kinova_cartesian_teleop")
    TeleopNode().run()


if __name__ == "__main__":
    main()