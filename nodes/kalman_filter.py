#!/usr/bin/env python3
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from hippo_msgs.msg import RangeMeasurement, RangeMeasurementArray
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from tf_transformations import euler_from_quaternion


class PositionKalmanFilter(Node):

    def __init__(self):
        # ---------------------
        # -- Initializadtion --
        # ---------------------
        super().__init__(node_name='position_kalman_filter')

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)

        self.init_params()

        # ---------------------
        # -- class variables --
        # ---------------------
        self.time_last_prediction = self.get_clock().now()

        # TODO Assuming state consists of position x,y,z -> Feel free to add
        # more!
        self.num_states = 3

        # initial state
        self.x0 = np.zeros((self.num_states, 1))

        # state, this will be updated in Kalman filter algorithm
        self.state = np.copy(self.x0)

        # initial state covariance - how sure are we about the state?
        # TODO initial state covariance is tuning knob
        # dimension: num states x num states
        # matrix needs to be positive definite and symmetric
        self.P0 = 0.1 * np.eye(self.num_states)

        # state covariance, this will be updated in Kalman filter algorithm
        self.P = self.P0

        # process noise covariance - how much noise do we add at each
        # prediction step?
        # TODO tuning knob
        # dimension: num states x num states
        # matrix needs to be positive definite and symmetric
        self.process_noise_position_stddev = 0.1
        self.Q = (self.range_noise_stddev**2) * np.eye(self.num_states)

        # measurement noise covariance - how much noise does the measurement
        # contain?
        # TODO tuning knob
        self.range_noise_stddev = 0.1
        # dimnesion: num measurements x num measurements
        # attention, this size is varying! -> Depends on detected Tags

        self.update_tag_poses()

        # --------------------------------
        # -- Publsihers and Subscribers --
        # --------------------------------
        self.position_pub = self.create_publisher(
            msg_type=PoseStamped,
            topic='position_estimate',
            qos_profile=1)
        
        self.ranges_sub = self.create_subscription(
            msg_type=RangeMeasurementArray,
            topic='ranges',
            callback=self.on_ranges,
            qos_profile=qos)
        self.vision_pose_sub = self.create_subscription(
            msg_type=PoseWithCovarianceStamped,
            topic='vision_pose_cov',
            callback=self.on_vision_pose,
            qos_profile=qos)
        
        # --------------------------------
        # -- timer: 50 times per second --
        # --------------------------------
        self.process_update_timer = self.create_timer(
            1.0 / 50, self.on_prediction_step_timer)

    def init_params(self) -> None:
        self.declare_parameters(
            namespace='',
            parameters=[
                ('range_noise_stddev', rclpy.Parameter.Type.DOUBLE),
                ('process_noise_position_stddev', rclpy.Parameter.Type.DOUBLE),
                ('tag_0_pos.x1', rclpy.Parameter.Type.DOUBLE),
                ('tag_0_pos.x2', rclpy.Parameter.Type.DOUBLE),
                ('tag_0_pos.x3', rclpy.Parameter.Type.DOUBLE),
            ]
        )

        param = self.get_parameter('range_noise_stddev')
        self.get_logger().info(f'{param.name}={param.value}')
        self.range_noise_stddev = param.value

        param = self.get_parameter('process_noise_position_stddev')
        self.get_logger().info(f'{param.name}={param.value}')
        self.process_noise_position_stddev = param.value

        # ------------------------
        # ---- Tag 0 Position ----
        self.tag_0_pos = []

        param = self.get_parameter('tag_0_pos.x1')
        self.get_logger().info(f'{param.name}={param.value}')
        self.tag_0_pos.append(param.value)

        param = self.get_parameter('tag_0_pos.x2')
        self.get_logger().info(f'{param.name}={param.value}')
        self.tag_0_pos.append(param.value)

        param = self.get_parameter('tag_0_pos.x3')
        self.get_logger().info(f'{param.name}={param.value}')
        self.tag_0_pos.append(param.value)


    def update_tag_poses(self) -> None:
        self.tag_poses = np.array([self.tag_0_pos, self.tag_0_pos,
                                   self.tag_0_pos, self.tag_0_pos])       
        tag_offsets = np.array([[0.0, 0.0, 0.0], [0.0, 0.6, 0.0],
                                [0.0, 0.0, -0.4], [0.0, 0.6, -0.4]])
        
        for count, (pos, offset) in enumerate(zip(self.tag_poses, tag_offsets)):
            self.tag_poses[count] = pos + offset
        
        self.get_logger().info(f'------ New Tag Positions ------')
        for count, position in enumerate(self.tag_poses):
            self.get_logger().info(f'Tag {count}: ({position[0]}|{position[1]}|{position[2]})')
        self.get_logger().info(f'-------------------------------')


    def on_params_changed(self, params) -> SetParametersResult:
        param: rclpy.Parameter
        for param in params:
            self.get_logger().info(f'Try to set [{param.name}] = {param.value}')
            if param.name == 'range_noise_stddev':
                self.range_noise_stddev = param.value
            elif param.name == 'process_noise_position_stddev':
                self.process_noise_position_stddev = param.value
                self.Q = (self.process_noise_position_stddev**2) * np.eye(
                    self.num_states)
            elif param.name == 'tag_0_pos.x1':
                self.tag_0_pos[0] = param.value
                self.update_tag_poses()
            elif param.name == 'tag_0_pos.x2':
                self.tag_0_pos[1] = param.value
                self.update_tag_poses()
            elif param.name == 'tag_0_pos.x3':
                self.tag_0_pos[2] = param.value
                self.update_tag_poses()
            else:
                continue
        return SetParametersResult(successful=True, reason='Parameter set')


    def on_ranges(self, ranges_msg: RangeMeasurementArray) -> None:
        # how many tags are detected?
        num_measurements = len(ranges_msg._measurements)

        # if no tags are detected, stop here
        if not num_measurements:
            return

        def get_jacobian_H (ranges_msg: RangeMeasurementArray):
            H = np.array([[], [], []])
            measurement_j: RangeMeasurement
            for index, measurement_j in enumerate(ranges_msg.measurements):
                H_i = np.array([])
                for pos_j, tag_pos_ij in zip(self.state[0:3], self.tag_poses[measurement_j.id]):
                    numerator = pos_j - tag_pos_ij
                    denumerator = 0.0
                    for pos_k, tag_pos_ik in zip(self.state[0:3], self.tag_poses[measurement_j.id]):
                        denumerator += np.power(pos_k - tag_pos_ik, 2)
                    denumerator = np.sqrt(denumerator)
                    H_i = np.append(H_i, numerator/denumerator)
                H = np.append(H, H_i, axis=1)
            return np.reshape(H, (-1,3))

        H = get_jacobian_H(ranges_msg=ranges_msg)
        

        # before the measurement update, let's do a process update
        now = self.get_clock().now()
        dt = (now - self.time_last_prediction).nanoseconds * 1e-9
        self.prediction(dt)
        self.time_last_prediction = now

        # TODO
        # self.measurement_update(...)


    def on_vision_pose(self, msg: PoseWithCovarianceStamped):
        # You might want to consider the vehicle's orientation

        # get the vehicle orientation expressed as quaternion
        q = msg.pose.pose.orientation
        # convert quaternion to euler angles
        (roll, pitch, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # TODO


    def on_prediction_step_timer(self) -> None:
        # We will do a prediction step with a constant rate
        now = self.get_clock().now()
        dt = (now - self.time_last_prediction).nanoseconds * 1e-9

        self.prediction(dt)
        self.time_last_prediction = now

        # publish the estimated pose with constant rate
        self.publish_pose_msg(state=np.copy(self.state), now=now)


    def measurement_update(self):
        vehicle_position = np.copy(self.state[0:3, 0])
        # TODO
        pass


    def prediction(self, dt: float) -> None:
        A = np.eye(self.num_states)
        self.state = A @ self.state
        self.P = A @ self.P @ A.transpose() + self.Q


    def publish_pose_msg(self, state: np.ndarray, now: rclpy.time.Time) -> None:
        msg = PoseStamped()

        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = state[0, 0]
        msg.pose.position.y = state[1, 0]
        msg.pose.position.z = state[2, 0]

        self.position_pub.publish(msg)


def main():
    rclpy.init()
    node = PositionKalmanFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
    