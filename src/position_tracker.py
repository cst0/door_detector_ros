#!/usr/bin/env python3

from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Transform
from image_geometry import PinholeCameraModel
import rospy
import cv2
import tf2_ros
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from door_detector_ros.srv import (
    DoorDetection,
)

"""
Subscribe to the detection topics, rectified camera Image topic, and depth Image topic.
When a detection is provided on the detection topic, find the corresponding point
in the Depth image and convert it to a 3D point. Track those points in the world
frame.
"""


class PositionTracker:
    def __init__(self):
        self.camera_model = PinholeCameraModel()
        self.rgb_image_sub = rospy.Subscriber(
            "/spot/camera/hand_color/image", Image, self.rgb_image_callback, queue_size=1
        )
        self.rgb_info_sub = rospy.Subscriber(
            "/spot/camera/hand_color/camera_info", CameraInfo, self.rgb_info_callback, queue_size=1
        )
        self.depth_image_subs = rospy.Subscriber(
            "/spot/camera/hand_image/image", Image, self.depth_image_callback, queue_size=1
        )
        self.detection_subs = rospy.Subscriber(
            "/door_detections", Detection2D, self.detection_callback, queue_size=1
        )

        self.handle_markers_pub = rospy.Publisher(
            "/handle_markers", MarkerArray, queue_size=1
        )
        self.hinge_markers_pub = rospy.Publisher(
            "/hinge_markers", MarkerArray, queue_size=1
        )

        self.cv_bridge = CvBridge()

        self.tracked_hinges = []
        self.tracked_handles = []
        self.rgb_image = None
        self.depth_image = None

        self.door_detection_service_provider = rospy.Service(
            "door_detection", DoorDetection, self.handle_door_detection_srv
        )

        # construct a tf buffer to use later
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def scale_image_to_image(self, image_transform, image_reference):
        image_refererence_x_scale = image_reference.shape[1] / image_transform.shape[1]
        image_refererence_y_scale = image_reference.shape[0] / image_transform.shape[0]
        return cv2.resize(image_transform, None, fx=image_refererence_x_scale, fy=image_refererence_y_scale, interpolation=cv2.INTER_CUBIC)

    def handle_door_detection_srv(self, _):
        pass

    def rgb_image_callback(self, msg):
        self.rgb_image = self.cv_bridge.imgmsg_to_cv2(msg)

    def rgb_info_callback(self, msg):
        self.camera_model.fromCameraInfo(msg)

    def depth_image_callback(self, msg):
        depth_image = self.cv_bridge.imgmsg_to_cv2(msg)
        if self.rgb_image is not None:
            self.depth_image = self.scale_image_to_image(depth_image, self.rgb_image)

    def find_depth_point(self, detection: Point):
        # from the x and y in the rgb image, return the corresponding point in the depth image
        if self.depth_image is None:
            rospy.logerr_throttle(1, "No depth image received, cannot find depth point")
            return Point()

        return Point(
            detection.x,
            detection.y,
            self.depth_image[int(detection.y)][int(detection.x)]/1000,
        )

    def depth_to_point(self, depth_point, camera_frame):
        x = depth_point.x
        y = depth_point.y
        depth = depth_point.z

        try:
            # use ros image geometry to convert the pixel coordinates to a 3D point
            point = self.camera_model.projectPixelTo3dRay(
                    (x, y)
                )
        except TypeError as e:
            rospy.logerr_throttle(1, "caught TypeError in projectPixelTo3dRay, {} {}".format((x, y, depth), e))
            return Point()

        # multiply the point by the depth to get the 3D point in the camera frame
        point = Point(
            point[0] * depth,
            point[1] * depth,
            point[2] * depth)

        # transform the point to the camera frame
        try:
            transform = self.tf_buffer.lookup_transform(
                camera_frame, "odom", rospy.Time(0), rospy.Duration(1.0)
            )
            tf = Transform(translation=point)
            point = np.dot(transform, tf)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr("Could not transform point")
            return Point()

        return Point(tf.transform.translation.x, tf.transform.translation.y, point.z)


    def append_door_tracker(self, point):
        self.tracked_hinges.append(point)
        if len(self.tracked_hinges) > 10:
            self.tracked_hinges.pop(0)

        self.hinge_markers_pub.publish(
            self.create_markers(self.tracked_hinges, "hinge", 1, Marker.CUBE)
        )

    def append_handle_tracker(self, point):
        self.tracked_handles.append(point)
        if len(self.tracked_handles) > 10:
            self.tracked_handles.pop(0)

        self.handle_markers_pub.publish(
            self.create_markers(self.tracked_handles, "handle", 2, Marker.SPHERE)
        )

    def create_markers(self, points, label: str, id_, type_):
        markers = MarkerArray()
        markers.markers = []
        for point in points:
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = label
            marker.id = id_
            marker.type = type_
            marker.action = Marker.ADD
            marker.pose.position = point
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = rospy.Duration(0.5)
            markers.markers.append(marker)
        return markers

    def detection_callback(self, msg: Detection2D):
        if msg.results:
            for detection in msg.results:
                detection:ObjectHypothesisWithPose
                if detection.id == 0:
                    # find the depth point corresponding to the detection
                    depth_point = self.find_depth_point(detection.pose.pose.position)
                    # convert the depth point to a 3D point
                    point = self.depth_to_point(depth_point, msg.header.frame_id)
                    # publish the point
                    self.append_door_tracker(point)
                if detection.id == 1:
                    depth_point = self.find_depth_point(detection.pose.pose.position)
                    point = self.depth_to_point(depth_point, msg.header.frame_id)
                    self.append_handle_tracker(point)

if __name__ == '__main__':
    rospy.init_node('position_tracker')
    position_tracker = PositionTracker()
    rospy.spin()
    rospy.loginfo("Shutting down position tracker")
    rospy.signal_shutdown("Shutting down position tracker")
