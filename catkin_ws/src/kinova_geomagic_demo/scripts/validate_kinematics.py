#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf2_ros
import tf.transformations as tr
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Vector3Stamped
from std_msgs.msg import Float64, Int32MultiArray

# Matriz de alinhamento Stylus -> Kinova EE definida no script de controle
EE_R_H_OFFSET = np.array([
    [ 0.0, 1.0,  0.0],
    [ 1.0, 0.0,  0.0],
    [ 0.0, 0.0, -1.0]
])

def dh_matrix(theta, d, a, alpha):
    """Calcula a matriz homogênea analítica via convenção DH."""
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def compute_fk(q, dh_params, base_transform=np.eye(4)):
    """Computa a Cinemática Direta analítica."""
    T = base_transform.copy()
    for i in range(len(q)):
        theta_offset, d, a, alpha = dh_params[i]
        theta_total = q[i] + theta_offset
        T = T @ dh_matrix(theta_total, d, a, alpha)
    return T

def rot_to_euler_parameters(R):
    """Algoritmo de Markley (2008) para extração de parâmetros de Euler."""
    trR = np.trace(R)
    candidates = np.array([R[0, 0], R[1, 1], R[2, 2], trR])
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
        
    q = x / np.linalg.norm(x)
    return q[3], q[0:3]  # eta, eps

def compute_orientation_error_euler(R_d, R_e):
    """Calcula o vetor de erro de rotação (Caccavale et al.) projetado na base."""
    R_de = R_e.T @ R_d
    eta_de, eps_de_e = rot_to_euler_parameters(R_de)
    if eta_de < 0.0:
        eps_de_e = -eps_de_e
    return R_e @ eps_de_e


class TeleopDiagnosticsNode:
    def __init__(self):
        rospy.init_node("teleop_diagnostics_node")

        # Parâmetros DH do Geomagic Touch para o frame 'stylus' (d6 = 0.0)
        self.hap_dh_params = [
            (0.0,       0.125,   0.0,   np.pi / 2),
            (0.0,       0.0,     0.14,  0.0),
            (0.0,       0.0,     0.0,   np.pi / 2),
            (0.0,       0.1325,  0.0,  -np.pi / 2),
            (np.pi / 2, 0.0,     0.0,  -np.pi / 2),
            (0.0,       0.0,     0.0,   0.0)
        ]
        self.hap_joint_names = ['waist', 'shoulder', 'elbow', 'yaw', 'pitch', 'roll']

        # Parâmetros DH do Kinova Gen3 7DoF
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

        self.workspace_scale = rospy.get_param("~workspace_scale", 2.0)
        self.kinova_base_frame = rospy.get_param("~kinova_base_frame", "base_link")
        self.geomagic_base_frame = rospy.get_param("~geomagic_base_frame", "base")

        # Variáveis de Estado
        self.enabled = False
        self._q_arm = None
        self.R_map = np.eye(3)
        self._hap0 = None
        self._ee0 = None
        self._R_hap0 = None
        self._R_ee0 = None

        # Publicadores
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.err_rot_pub = rospy.Publisher("/teleop/orientation_error", Vector3Stamped, queue_size=1)
        self.err_norm_pub = rospy.Publisher("/teleop/orientation_error_norm", Float64, queue_size=1)

        # Inscrições nos tópicos
        rospy.Subscriber("/geomagic/button", Int32MultiArray, self._on_button, queue_size=1)
        rospy.Subscriber("/my_gen3/joint_states", JointState, self._on_arm_state, queue_size=1)
        rospy.Subscriber("/geomagic/joint_states", JointState, self._on_haptic_state, queue_size=1)

        rospy.loginfo("Nó teleop_diagnostics ativo. Publicando TFs e tópicos de erro de orientação.")

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

    def _on_button(self, msg):
        if not msg.data or len(msg.data) < 2:
            return
        if msg.data[0] == 1 and msg.data[1] == 0:
            self.enabled = True
            self._hap0 = None
            self._ee0 = None
            self._R_hap0 = None
            self._R_ee0 = None
        elif msg.data[1] == 1 and msg.data[0] == 0:
            self.enabled = False
            self._hap0 = None
            self._ee0 = None
            self._R_hap0 = None
            self._R_ee0 = None

    def _on_arm_state(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        self._q_arm = np.array([name_to_pos.get(name, 0.0) for name in self.arm_joint_names])
        T_ee_base = compute_fk(self._q_arm, self.arm_dh_params, base_transform=self.arm_base_transform)
        self._broadcast_tf(T_ee_base, self.kinova_base_frame, "calculated_kinova_ee")

    def _on_haptic_state(self, msg):
        if self._q_arm is None:
            return

        name_to_pos = dict(zip(msg.name, msg.position))
        q_h = np.array([name_to_pos.get(name, 0.0) for name in self.hap_joint_names])

        # FK Geomagic Touch (no frame stylus)
        T_h = compute_fk(q_h, self.hap_dh_params)
        p = T_h[0:3, 3]
        R_h = T_h[0:3, 0:3]
        self._broadcast_tf(T_h, self.geomagic_base_frame, "calculated_stylus")

        # FK Kinova Atual
        Tc = compute_fk(self._q_arm, self.arm_dh_params, base_transform=self.arm_base_transform)
        p_cur = Tc[0:3, 3]
        R_cur = Tc[0:3, 0:3]

        if not self.enabled:
            self._hap0 = None
            self._ee0 = None
            self._R_hap0 = None
            self._R_ee0 = None
            return

        # Guarda e inicialização da pose neutra
        if self._hap0 is None or self._ee0 is None:
            self._hap0 = p.copy()
            self._R_hap0 = R_h.copy()
            self._ee0 = p_cur.copy()
            self._R_ee0 = R_cur.copy()
            return

        # Projeção da Pose Desejada (idêntica ao script de controle)
        delta_pos = p - self._hap0
        p_des = self._ee0 + self.workspace_scale * delta_pos

        R_des = self.R_map @ R_h @ EE_R_H_OFFSET
        # Montagem e publicação do frame desejado no TF
        T_des = np.eye(4)
        T_des[0:3, 0:3] = R_des
        T_des[0:3, 3] = p_des
        self._broadcast_tf(T_des, self.kinova_base_frame, "target_ee_des")

        # Cálculo do erro de orientação exato usado no controle
        err_R = compute_orientation_error_euler(R_des, R_cur)
        err_R_norm = float(np.linalg.norm(err_R))

        # Publicação para telemetria e plot
        msg_vec = Vector3Stamped()
        msg_vec.header.stamp = rospy.Time.now()
        msg_vec.header.frame_id = self.kinova_base_frame
        msg_vec.vector.x = float(err_R[0])
        msg_vec.vector.y = float(err_R[1])
        msg_vec.vector.z = float(err_R[2])
        self.err_rot_pub.publish(msg_vec)

        self.err_norm_pub.publish(Float64(data=err_R_norm))


if __name__ == "__main__":
    try:
        TeleopDiagnosticsNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass