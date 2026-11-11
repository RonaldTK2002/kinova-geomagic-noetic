#!/usr/bin/env python3
import rospy
from kortex_driver.srv import ReadAction, ReadActionRequest, ExecuteAction, ExecuteActionRequest

def send_robot_home(robot_ns="my_gen3"):
    read_srv_name = f"/{robot_ns}/base/read_action"
    exec_srv_name = f"/{robot_ns}/base/execute_action"

    rospy.loginfo("Aguardando serviços de ação da base do Kinova...")
    rospy.wait_for_service(read_srv_name, timeout=10.0)
    rospy.wait_for_service(exec_srv_name, timeout=10.0)

    read_srv = rospy.ServiceProxy(read_srv_name, ReadAction)
    exec_srv = rospy.ServiceProxy(exec_srv_name, ExecuteAction)

    # 1. Lê a ação 'Home' registrada no robô (ID 2)
    req_read = ReadActionRequest()
    req_read.input.identifier = 2
    res_read = read_srv(req_read)

    # 2. Executa a ação lida
    req_exec = ExecuteActionRequest()
    req_exec.input = res_read.output

    rospy.loginfo("Enviando Kinova Gen3 para pose Home...")
    exec_srv(req_exec)
    rospy.loginfo("Comando de Home enviado com sucesso!")

if __name__ == "__main__":
    rospy.init_node("send_kinova_home_node")
    try:
        send_robot_home()
    except rospy.ROSInterruptException:
        pass