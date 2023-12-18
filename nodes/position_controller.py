#!/usr/bin/env python3
import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from hippo_msgs.msg import ActuatorSetpoint
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

class PositionControllerNode(Node):
    def __init__(self):
        # --------------------
        # -- Initialisation --
        # --------------------
        super().__init__(node_name='position_controller')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)

        self.init_params()
        self.add_on_set_parameters_callback(self.on_params_changed)
        
        # ---------------------
        # -- class variables --
        # ---------------------
        self.LOGGER_FILTERED_POSE = True    # <- toggles Publisher
        self.LOGGER_PID_CONTROLLER = False

        self.last_time = self.get_clock().now()
        self.last_error = np.zeros((3, 1))

        self.i_error = np.zeros((3, 1))

        self.filtered_pose = np.zeros((3, 1))
        self.setpoint_pose = np.zeros((3, 1))

        # -- Transformation Matrix --
        # - Tank coordinates to robot coordinates -
        self.T = np.array([
            +0.0, +1.0, +0.0,
            -1.0, +0.0, +0.0,
            +0.0, +0.0, +1.0,
        ]).reshape((3, 3))

        # --------------------------------
        # -- Publishers and Subscribers --
        # --------------------------------
        self.pose_and_cov_estimate_sub = self.create_subscription(
            msg_type=PoseWithCovarianceStamped,
            topic='pose_and_cov_estimate',
            callback=self.on_position,
            qos_profile=qos)
        self.pose_setpoint_sub = self.create_subscription(
            msg_type=PoseStamped,
            topic='setpoint_pose',
            callback=self.on_setpoint,
            qos_profile=qos)
        
        self.filtered_pose_pub = self.create_publisher(
            msg_type=PoseStamped,
            topic='filtered_pose',
            qos_profile=1)
        self.thrust_pub = self.create_publisher(
            msg_type=ActuatorSetpoint,
            topic='thrust_setpoint',
            qos_profile=1)
        

    # ----------------------------------------------
    # ---------- Initializadtion & Update ----------
    def init_params(self) -> None:
        self.declare_parameters(
            namespace='',
            parameters=[
                ('moving_average_beta', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('sigma_shutoff_value', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('i_controller_shutoff_error', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('gains.p', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('gains.i', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('gains.d', rclpy.Parameter.Type.DOUBLE_ARRAY),
            ]
        )

        param = self.get_parameter('moving_average_beta')
        self.get_logger().info(f'{param.name}={param.value}')
        self.moving_average_beta = np.array(param.value).reshape((3, 1))

        param = self.get_parameter('sigma_shutoff_value')
        self.get_logger().info(f'{param.name}={param.value}')
        self.sigma_shutoff_value = np.array(param.value).reshape((3, 1))

        param = self.get_parameter('i_controller_shutoff_error')
        self.get_logger().info(f'{param.name}={param.value}')
        self.i_controller_shutoff_error = np.array(param.value).reshape((3, 1))

        param = self.get_parameter('gains.p')
        self.get_logger().info(f'{param.name}={param.value}')
        self.p_gain = np.array(param.value).reshape((3, 1))

        param = self.get_parameter('gains.i')
        self.get_logger().info(f'{param.name}={param.value}')
        self.i_gain = np.array(param.value).reshape((3, 1))

        param = self.get_parameter('gains.d')
        self.get_logger().info(f'{param.name}={param.value}')
        self.d_gain = np.array(param.value).reshape((3, 1))


    def on_params_changed(self, params) -> SetParametersResult:
        param: rclpy.Parameter
        for param in params:
            if   param.name == 'moving_average_beta':
                self.moving_average_beta = np.array(param.value).reshape((3, 1))
            elif param.name == 'sigma_shutoff_value':
                self.sigma_shutoff_value = np.array(param.value).reshape((3, 1))
            elif param.name == 'i_controller_shutoff_error':
                self.i_controller_shutoff_error = np.array(param.value).reshape((3, 1))
            elif param.name == 'gains.p':
                self.p_gain = np.array(param.value).reshape((3, 1))
                self.get_logger().info(f'P gain = {self.p_gain.flatten()}')
            elif param.name == 'gains.i':
                self.i_gain = np.array(param.value).reshape((3, 1))
                self.i_error = np.zeros((3, 1))
                self.get_logger().info(f'I gain = {self.i_gain.flatten()}')
            elif param.name == 'gains.d':
                self.d_gain = np.array(param.value).reshape((3, 1))
                self.get_logger().info(f'D gain = {self.d_gain.flatten()}')
            else:
                continue
        return SetParametersResult(successful=True, reason='Parameter set')

    # ---------------------------------
    # ---------- Subscribers ----------
    def on_position(self, msg: PoseWithCovarianceStamped) -> None:
        pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ]).reshape((3, 1))

        covariance = np.array(msg.pose.covariance).reshape((6, 6))

        now = self.get_clock().now()

        self._moving_average_filter(pose=np.copy(pose), beta=np.copy(self.moving_average_beta))
        if self.LOGGER_FILTERED_POSE:
            self.publish_filtered_pose(filtered_pose=np.copy(self.filtered_pose), now=now)

        thrust = self.compute_control_output(covariance=covariance)
        self.publish_thrust(thrust_tank=thrust, timestamp=now)


    def on_setpoint(self, msg: PoseStamped) -> None:
        self.setpoint_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]).reshape((3, 1))

    # ----------------------------------
    # ---------- Calculations ----------
    def _moving_average_filter(self, pose: np.ndarray, beta: np.ndarray) -> None:
        self.filtered_pose = np.multiply(pose, beta) + np.multiply((1-beta), self.filtered_pose)


    def compute_control_output(self, covariance: np.ndarray) -> np.ndarray:        
        error = self.setpoint_pose - self.filtered_pose
        d_error = error - self.last_error
        average_error = np.divide(error + self.last_error, 2.0) 
        self.last_error = error

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        
        for i in range(len(self.i_error)):
            if dt > 0.2 or abs(error[i, 0]) > self.i_controller_shutoff_error[i, 0]:
                self.i_error[i, 0] = 0.0
            else:
                self.i_error[i, 0] += average_error[i, 0] * dt

        p_thrust = np.multiply(self.p_gain, error)
        d_thrust = np.multiply(self.d_gain, np.divide(d_error, dt))
        i_thrust = np.multiply(self.i_gain, self.i_error)

        thrust = p_thrust + d_thrust + i_thrust

        if self.LOGGER_PID_CONTROLLER:
            self.get_logger().info(f'P Controller: {p_thrust.flatten()}')
            self.get_logger().info(f'I Controller: {i_thrust.flatten()}')
            self.get_logger().info(f'D Controller: {d_thrust.flatten()}')
            self.get_logger().info(f'PID Controller: {thrust.flatten()}')

        # -- Check if out of bounds --
        if self.filtered_pose[0, 0] > 1.5:
            thrust[0, 0] = -0.1
            self.i_error[0, 0] = 0.0
        elif self.filtered_pose[0, 0] < 0.5:
            thrust[0, 0] = 0.1
            self.i_error[0, 0] = 0.0

        if self.filtered_pose[1, 0] > 3.5:
            thrust[1, 0] = -0.1
            self.i_error[1, 0] = 0.0
        elif self.filtered_pose[1, 0] < 1.5:
            thrust[1, 0] = 0.1
            self.i_error[1, 0] = 0.0

        if self.filtered_pose[2, 0] > -0.1:
            thrust[2, 0] = -0.1
            self.i_error[2, 0] = 0.0
        elif self.filtered_pose[2, 0] < -0.8:
            thrust[2, 0] = 0.1
            self.i_error[2, 0] = 0.0

        # -- Check if between +/- one --
        for i in range(len(thrust)):
            if thrust[i] > 1.0:
                thrust[i] = 1.0
            elif thrust[i] < -1.0:
                thrust[i] = -1.0

        # -- Check if covariance low enough --
        for i in range(len(thrust)):
            if covariance[i, i] > np.power(self.sigma_shutoff_value[i, 0], 2):
                thrust[i, 0] = 0.0
                self.i_error[i, 0] = 0.0

        if self.LOGGER_PID_CONTROLLER:
            self.get_logger().info(f'final PID output: {thrust.flatten()}')

        return thrust

    # ------------------------------------
    # ---------- Publish values ----------
    def publish_filtered_pose(self, filtered_pose: np.ndarray, now: rclpy.time.Time) -> None:
        msg = PoseStamped()
        msg.header.stamp = now.to_msg()

        msg.pose.position.x = filtered_pose[0, 0]
        msg.pose.position.y = filtered_pose[1, 0]
        msg.pose.position.z = filtered_pose[2, 0]

        self.filtered_pose_pub.publish(msg=msg)
    
    
    def publish_thrust(self, thrust_tank: np.ndarray, timestamp: rclpy.time.Time) -> None:
        msg = ActuatorSetpoint()
        msg.header.stamp = timestamp.to_msg()

        # -- tank thrust vector -> robot thrust vector --
        robot_thrust = self.T @ thrust_tank

        msg.ignore_x = False
        msg.ignore_y = False
        msg.ignore_z = False

        msg.x = robot_thrust[0, 0]
        msg.y = robot_thrust[1, 0]
        msg.z = robot_thrust[2, 0]

        self.thrust_pub.publish(msg)


def main():
    rclpy.init()
    node = PositionControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()