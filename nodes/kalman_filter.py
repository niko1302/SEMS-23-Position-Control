#!/usr/bin/env python3
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from hippo_msgs.msg import RangeMeasurement, RangeMeasurementArray
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from tf_transformations import euler_from_quaternion

from custom_msgs.msg import CovarianceDetP


class PositionKalmanFilter(Node):

    def __init__(self):
        # --------------------
        # -- Initialisation --
        # --------------------
        super().__init__(node_name='position_kalman_filter')

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)

        self.init_params()

        self.update_tag_poses()

        # ---------------------
        # -- class variables --
        # ---------------------
        self.DEBUG_LOGGER_STATE_AND_COVARIANCE = False
        self.DEBUG_LOGGER_KALMAN_GAIN_CALCULATION = False
        
        # ---- Sigmas ----
        # self.sigma_prediction_pos --> Parameter
        # self.sigma_prediction_vel --> Parameter
        # self.sigma_measurement --> Parameter

        sigma_initial_pos = 0.05   # [m] for P0
        sigma_initial_vel = 0.5     # [m/s] foor P0

        # ---- Dimensions ----
        self.num_dimensions = 3     # (x1|x2|x3) or (x|y|z)
        self.num_derivations = 2    # pos, vel
        self.num_states = self.num_dimensions * self.num_derivations    # total states

        # ---- State x ----
        pos0 = [+1.3, +1.0, -0.5]
        vel0 = [+0.0, +0.0, +0.1]
        x0 = np.array([pos0, vel0])
        self.x = x0.reshape((self.num_states, 1))
        
        # ---- standart deviation P ----
        self.P = np.eye(self.num_states)
        self.P[:, 0:3] *= np.power(sigma_initial_pos, 2)
        self.P[:, 3:6] *= np.power(sigma_initial_vel, 2)

        # ---- standart deviation gain per prediction Q ----
        self.Q = np.eye(self.num_states)
        self.Q[:, 0:3] *= np.power(self.sigma_prediction_pos, 2)
        self.Q[:, 3:6] *= np.power(self.sigma_prediction_vel, 2)
        
        # -----------------------------------------------------
        self.time_last_prediction = self.get_clock().now()

        self.add_on_set_parameters_callback(self.on_params_changed)

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

        self.covariance_pub = self.create_publisher(
            msg_type=CovarianceDetP,
            topic='covaricance',
            qos_profile=1)
        
        # --------------------------------
        # -- timer: 50 times per second --
        # --------------------------------
        self.process_update_timer = self.create_timer(
            1.0 / 50, self.on_prediction_step_timer)


    # ----------------------------------------------
    # ---------- Initializadtion & Update ----------
    def init_params(self) -> None:
        self.declare_parameters(
            namespace='',
            parameters=[
                ('sigma_prediction_pos', rclpy.Parameter.Type.DOUBLE),
                ('sigma_prediction_vel', rclpy.Parameter.Type.DOUBLE),
                ('sigma_measurement', rclpy.Parameter.Type.DOUBLE),
                ('offset_to_tag0_x1', rclpy.Parameter.Type.DOUBLE),
                ('offset_to_tag0_x3', rclpy.Parameter.Type.DOUBLE),
                ('tag_0_pos.x1', rclpy.Parameter.Type.DOUBLE),
                ('tag_0_pos.x2', rclpy.Parameter.Type.DOUBLE),
                ('tag_0_pos.x3', rclpy.Parameter.Type.DOUBLE),
            ]
        )

        # ----------------
        # ---- Sigmas ----

        # -- Sigma gain for the prediction --
        param = self.get_parameter('sigma_prediction_pos')
        self.get_logger().info(f'{param.name}={param.value}')
        self.sigma_prediction_pos = param.value

        param = self.get_parameter('sigma_prediction_vel')
        self.get_logger().info(f'{param.name}={param.value}')
        self.sigma_prediction_vel = param.value

        # -- Sigma gain for the measurement --
        param = self.get_parameter('sigma_measurement')
        self.get_logger().info(f'{param.name}={param.value}')
        self.sigma_measurement = param.value

        # -- Tag offset --
        param = self.get_parameter('offset_to_tag0_x1')
        self.get_logger().info(f'{param.name}={param.value}')
        self.offset_to_tag0_x1 = param.value

        param = self.get_parameter('offset_to_tag0_x3')
        self.get_logger().info(f'{param.name}={param.value}')
        self.offset_to_tag0_x3 = param.value

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
        tag_offsets = np.array([[0.0, 0.0, 0.0], [self.offset_to_tag0_x1, 0.0, 0.0],
                                [0.0, 0.0, self.offset_to_tag0_x3], [self.offset_to_tag0_x1, 0.0, self.offset_to_tag0_x3]])
        
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
            if   param.name == 'sigma_prediction_pos':
                self.sigma_prediction_pos = param.value
                self.Q[:, 0:3] *= np.power(self.sigma_prediction_pos, 2)
            elif param.name == 'sigma_prediction_vel':
                self.sigma_prediction_vel = param.value
                self.Q[:, 3:6] *= np.power(self.sigma_prediction_vel, 2)
            
            elif param.name == 'sigma_measurement':
                self.sigma_measurement = param.value

            elif param.name == 'offset_to_tag0_x1':
                self.offset_to_tag0_x1 = param.value
                self.update_tag_poses()
            elif param.name == 'offset_to_tag0_x3':
                self.offset_to_tag0_x3 = param.value
                self.update_tag_poses()
            
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

    # -------------------------------
    # ---------- Do Things ----------
    def on_ranges(self, ranges_msg: RangeMeasurementArray) -> None:
        # how many tags are detected?
        num_measurements = len(ranges_msg._measurements)

        # if no tags are detected, stop here
        if not num_measurements:
            return

        self.measurement_update(ranges_msg=ranges_msg, num_measurements=num_measurements)
        
        now = self.get_clock().now()
        self.publish_cov(predict_or_measure=0.0, now=now)
        self.publish_pose_msg(state=np.copy(self.x), now=now)
        

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

        self.publish_cov(predict_or_measure=1.0, now=now)

        # publish the estimated pose with constant rate
        self.publish_pose_msg(state=np.copy(self.x), now=now)

    # -----------------------------------
    # ---------- Kalman Filter ----------
    def measurement_update(self, ranges_msg: RangeMeasurementArray, num_measurements: int) -> None:
        measurement: RangeMeasurement

        # -- Calculate measurement residual --
        # -> the difference between our measurement and what the measurement should be (depends on where we think we are)
        y = np.array([])
        for measurement in ranges_msg.measurements:
            z_k = measurement.range
            hx = self._get_dist_between_vec(self.x[0:3, 0], self.tag_poses[measurement.id])
            y = np.append(y, z_k - hx)
        y = y.reshape((num_measurements, 1))

        # -- Get Jacobi Matrix H --
        H = self._get_ranges_jacobian_H(ranges_msg=ranges_msg)

        # -- Get Kalman gain matrix K --
        R = np.power(self.sigma_measurement, 2) * np.eye(num_measurements)
        S = H @ self.P @ H.transpose() + R
        K = self.P @ H.transpose() @ np.linalg.inv(S)

        # -- debug logger --
        if self.DEBUG_LOGGER_KALMAN_GAIN_CALCULATION:
            self.get_logger().info(f'RANGES: y =\n{y}')
            self.get_logger().info(f'RANGES: H =\n{H}')
            self.get_logger().info(f'RANGES: K =\n{K}')

        # -- Predict one last time before our measurement update --
        now = self.get_clock().now()
        dt = (now - self.time_last_prediction).nanoseconds * 1e-9
        self.prediction(dt)
        self.time_last_prediction = now

        # -- Update state an covariance --
        self.x = self.x + (K @ y)
        self.P = (np.eye(self.num_states) -  (K @ H)) @ self.P

        # -- debug logger --
        if self.DEBUG_LOGGER_STATE_AND_COVARIANCE:
            self.get_logger().info(f'RANGES: x = {self.x.flatten()}')
            self.get_logger().info(f'RANGES: P = {np.linalg.det(self.P)}')


    def prediction(self, dt: float) -> None:
        A = self._get_A(dt=dt)
        self.x = A @ self.x
        self.P = A @ self.P @ A.transpose() + self.Q
        
        if self.DEBUG_LOGGER_STATE_AND_COVARIANCE:
            self.get_logger().info(f'Prediction: x = {self.x.flatten()}')
            self.get_logger().info(f'Prediction: P = {np.linalg.det(self.P)}')
    
    # -----------------------------------
    # ---------- get functions ----------
    def _get_A(self, dt:float) -> np.ndarray:
        A = np.eye(self.num_states)
        for m, n in zip(range(0,3), range(3,self.num_states)):
            A[m, n] = dt
        return A


    def _get_ranges_jacobian_H (self, ranges_msg: RangeMeasurementArray) -> np.ndarray:
        H = np.array([])
        measurement: RangeMeasurement
        for measurement in ranges_msg.measurements:
            dist = self._get_dist_between_vec(self.x, self.tag_poses[measurement.id])
            for x_j, p_j in zip(self.x[0:3, 0], self.tag_poses[measurement.id]):
                H = np.append(H, [(x_j - p_j) / dist])
            for i in range(3, self.num_states):
                H = np.append(H, [0.0])
        return H.reshape((-1, self.num_states))


    def _get_dist_between_vec(self, posA: np.ndarray, posB: np.ndarray) -> float:
        dist = 0.0
        for a, b in zip(posA, posB):
            dist += np.power(a-b, 2)
        return max(np.sqrt(dist), 1e-8)

    # ------------------------------------
    # ---------- Publish values ----------
    def publish_cov(self, predict_or_measure: float, now: rclpy.time.Time) -> None:
        msg = CovarianceDetP()
        msg.header.stamp = now.to_msg()
        msg.predict_or_measure = predict_or_measure
        msg.determinant_cov = np.linalg.det(self.P)
        self.covariance_pub.publish(msg)


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
    