#!/usr/bin/env python3

import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

class PositionControllerNode(Node):
    def __init__(self):
        # --------------------
        # -- Initialisation --
        # --------------------
        super().__init__(node_name='position_setpoint')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)

        self.init_params()
        self.add_on_set_parameters_callback(self.on_params_changed)

        # ---------------------
        # -- class variables --
        # ---------------------
        self.start_time =self.get_clock().now()

        self.room_origin = np.array([0.9, 2.5, -0.4]).reshape((3, 1))
        self.room_lenght = np.array([0.2, 0.2, -0.2]).reshape((3, 1))
        self.room_center = self.room_origin + np.divide(self.room_lenght, 2.0)

        # --------------------------------
        # -- Publishers and Subscribers --
        # --------------------------------
        self.setpoint_pose_pub = self.create_publisher(
            msg_type=PoseStamped,
            topic='setpoint_pose',
            qos_profile=1)

        # --------------------------------
        # -- timer: 50 times per second --
        # --------------------------------
        self.process_update_timer = self.create_timer(
            timer_period_sec=(1.0 / 50),
            callback=self.on_timer)

    # ----------------------------------------------
    # ---------- Initializadtion & Update ----------
    def init_params(self) -> None:
        self.declare_parameters(
            namespace='',
            parameters=[
                ('setpoint_function', rclpy.Parameter.Type.STRING),
                ('duration', rclpy.Parameter.Type.DOUBLE),
            ]
        )
        
        param = self.get_parameter('setpoint_function')
        self.get_logger().info(f'{param.name}={param.value}')
        self.setpoint_function = param.value

        param = self.get_parameter('duration')
        self.get_logger().info(f'{param.name}={param.value}')
        self.duration = param.value


    def on_params_changed(self, params) -> SetParametersResult:
        param: rclpy.Parameter
        for param in params:
            if param.name == 'setpoint_function':
                self.setpoint_function = param.value
            elif param.name == 'duration':
                self.duration = param.value
            else:
                continue
            
        return SetParametersResult(successful=True, reason='Parameter set')

    # --------------------------------------------
    # ---------- Subscribers and Timers ----------
    def on_timer(self) -> None:
        now  = self.get_clock().now()
        time = (now - self.start_time).nanoseconds * 1e-9

        if self.setpoint_function == 'rectangle':
            i = time % (self.duration * 4)
            setpoint = np.copy(self.room_origin)
            setpoint[2, 0] += (self.room_lenght[2, 0] / 2)
            if i >= 0 and i < (self.duration * 1.0):
                pass
            elif i >= (self.duration * 1.0) and i < (self.duration * 2.0):
                setpoint[0, 0] += self.room_lenght[0, 0]
            elif i >= (self.duration * 2.0) and i < (self.duration * 3.0):
                setpoint[0, 0] += self.room_lenght[0, 0]
                setpoint[1, 0] += self.room_lenght[1, 0]
            elif i >= (self.duration * 3.0) and i < (self.duration * 4.0):
                setpoint[1, 0] += self.room_lenght[1, 0]
        
        elif self.setpoint_function == 'circle':
            b = (np.pi) / self.duration
            setpoint = np.copy(self.room_center)
            setpoint += np.array([
                (self.room_lenght[0, 0] / 2) * np.cos(b * time),
                (self.room_lenght[1, 0] / 2) * np.sin(b * time),
                0
            ]).reshape((3, 1))

        elif self.setpoint_function == 'diagonal':
            i = time % (self.duration * 2)
            setpoint = np.copy(self.room_origin)

            if i >= 0 and i < self.duration:
                setpoint += self.room_lenght

        else:
            setpoint = np.copy(self.room_center)

        self.publish_setpoint(pose=setpoint, now=now)

    # ------------------------------------
    # ---------- Publish values ----------
    def publish_setpoint(self, pose: np.ndarray, now: rclpy.time.Time) -> None:
        msg = PoseStamped()
        msg.header.stamp = now.to_msg()

        msg.pose.position.x = pose[0, 0]
        msg.pose.position.y = pose[1, 0]
        msg.pose.position.z = pose[2, 0]

        self.setpoint_pose_pub.publish(msg=msg)


def main():
    rclpy.init()
    node = PositionControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()