#!/usr/bin/env python3

import argparse

import cv2
import numpy as np
import cv_bridge
import rospy
import rospkg

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
from spot_msgs.srv import DoorOpen, DoorOpenResponse, DoorOpenRequest
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

"""
Door detector class. Creates a ros Image subscriber with a callback
function that uses cv_bridge to convert the image to a numpy array. Weights are
passed into the readNet function to allow the door to be detected. Detections
are then published as a Detection2D ROS publisher.
"""

CLASSES = ["door", "handle", "cabinet", "refrigerator"]
DOOR_IDS = [0, 2, 3]
HANDLE_IDS = [1]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


class DoorDetector:
    def __init__(self, args):
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.image_callback, queue_size=1
        )
        self.image_pub = rospy.Publisher(
            "/camera/color/detections", Image, queue_size=1
        )
        self.depth_pub = rospy.Subscriber(
            "/camera/depth/image_rect_raw", Image, self.depth_callback, queue_size=1
        )
        self.detections_pub = rospy.Publisher(
            "/camera/detections", Detection2D, queue_size=1
        )
        self.open_door_srv = rospy.Service(
            "/run_open_door", Trigger, self.handle_open_door_callback
        )
        self.open_door_client = rospy.ServiceProxy("/open_door", DoorOpen)

        self.rotate_left = args.rotate_left
        self.rotate_right = args.rotate_right
        self.rotate_down = args.rotate_down

        # using the rospkg python api, we can find the location of the
        # door_detection package on the robot
        package_path = self.get_package_path()
        # the weights and config files are stored in the cfg folder
        weights_path = package_path + "/cfg/yolov3-door.weights"
        config_path = package_path + "/cfg/yolov3-door.cfg"

        self.net = cv2.dnn.readNet(
            weights_path,
            config_path,
        )

        self.raw_image = None
        self.depth_image = None
        self.detections = []
        self.detections_time = []
        self.tracking_length_seconds = 2
        self.timer_loop = rospy.Timer(rospy.Duration(0.5), self.timer_callback)

    def handle_open_door_callback(self, _):
        # we got a trigger here, so we're gonna figure out how to open the door
        req = DoorOpenRequest()
        if len(self.detections) == 0:
            rospy.logwarn("No detections, can't open door")
            return TriggerResponse(success=False, message="No detections")
        else:
            rospy.loginfo("Got some detections, opening door")
            try:
                req = self.populate_door_open_req(req)
            except Exception as e:
                rospy.logerr("Failed to populate door open request: {}".format(e))

        if req is None:
            return TriggerResponse(success=False, message="Couldn't open door")

        resp = self.open_door_client(req)
        return TriggerResponse(success=resp.success, message=resp.message)

    def populate_door_open_req(self, req):
        most_recent_detection: Detection2D = self.detections[-1]
        handle_position = None
        hinge_position = None

        assert most_recent_detection.results is not None, "No results in detection"
        assert self.raw_depth is not None, "No depth image"
        assert self.raw_image is not None, "No image"

        for result in most_recent_detection.results:
            if result.id in HANDLE_IDS:
                handle_position = result.pose.pose.position
        assert handle_position is not None, "No handle detected"

        for result in most_recent_detection.results:
            if result.id in DOOR_IDS:
                hinge_position = handle_position
                hinge_position.x = (
                    self.raw_image.shape[1] - handle_position.x
                )  # just drop it on the other side

        assert handle_position is not None, "No handle position found"
        assert hinge_position is not None, "No hinge position found"

        req.hinge_side = (
            req.HINGE_LEFT if hinge_position.x < handle_position.x else req.HINGE_RIGHT
        )
        # create a raycast of the handle position given the known x and y of the handle
        handle_x = handle_position.x
        handle_y = handle_position.y
        # get the depth at the handle position
        handle_depth = self.raw_depth[handle_y, handle_x]

        # now that we have x, y, and depth, we can convert it to a normalized vector
        handle_vector = np.array([handle_x, handle_y, handle_depth])
        handle_vector = handle_vector / np.linalg.norm(handle_vector)

        req.handle_raycast.x = handle_vector[0]
        req.handle_raycast.y = handle_vector[1]
        req.handle_raycast.z = handle_vector[2]

        return req

    def get_package_path(self, package_name="door_detector_ros"):
        rospack = rospkg.RosPack()
        return rospack.get_path(package_name)

    def timer_callback(self, event):
        del event
        if self.raw_image is None:
            return

        image = self.bridge.imgmsg_to_cv2(self.raw_image, "bgr8")
        self.raw_image = None

        # rotate image left
        if self.rotate_left:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # rotate image right
        if self.rotate_right:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # rotate image down
        if self.rotate_down:
            image = cv2.rotate(image, cv2.ROTATE_180)

        scale = 0.00392
        blob = cv2.dnn.blobFromImage(
            image, scale, (416, 416), (0, 0, 0), True, crop=False
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    width = int(detection[2] * image.shape[1])
                    height = int(detection[3] * image.shape[0])
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        conf_threshold = 0.2
        nms_threshold = 0.2

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            obj = ObjectHypothesisWithPose()
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            obj.pose.pose.position.x = x
            obj.pose.pose.position.y = y
            w = box[2]
            h = box[3]
            self.draw_prediction(
                image, class_ids[i], confidences[i], x, y, x + w, y + h
            )

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
        detections = self.get_detections(class_ids, confidences, boxes)
        self.detections_pub.publish(detections)

        # prune old detections
        self.detections.append(detections)
        self.detections_time.append(rospy.Time.now())
        for i in range(len(self.detections_time)):
            if rospy.Time.now() - self.detections_time[i] > rospy.Duration(
                self.tracking_length_seconds
            ):
                self.detections.pop(0)
                self.detections_time.pop(0)

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def get_detections(self, class_ids, confidences, boxes):
        detections = Detection2D()
        detections.header.stamp = rospy.Time.now()
        detections.header.frame_id = self.frame_id
        detections.results = []

        rospy.loginfo_throttle(5, "Got some detections: {}".format(boxes))
        for i in range(len(class_ids)):
            rospy.loginfo(
                "Class: %s, Confidence: %s", CLASSES[class_ids[i]], confidences[i]
            )
            obj = ObjectHypothesisWithPose()
            obj.id = class_ids[i]
            obj.score = confidences[i]
            obj.pose.pose.position.x = (boxes[i][0] + boxes[i][2]) / 2
            obj.pose.pose.position.y = (boxes[i][1] + boxes[i][3]) / 2

            detections.results.append(obj)

        return detections

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(CLASSES[class_id])
        confidence = "{:.2f}".format(confidence)
        color = COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(
            img,
            label + " " + confidence,
            (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    def depth_callback(self, data):
        try:
            self.raw_depth = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except cv_bridge.CvBridgeError as e:
            rospy.logerr(e)

    def image_callback(self, data):
        try:
            self.raw_image = data
            self.frame_id = data.header.frame_id
        except cv_bridge.CvBridgeError as e:
            rospy.logerr(e)


if __name__ == "__main__":
    # argument parsing: create argparse parser, parse for rotation-indicator flag
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotate_left", "-l", action="store_true")
    parser.add_argument("--rotate_right", "-r", action="store_true")
    parser.add_argument("--rotate_down", "-d", action="store_true")

    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("door_detector", anonymous=True)
    DoorDetector(args)
    rospy.spin()
