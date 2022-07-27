#!/usr/bin/env python3

import argparse

import cv2
import numpy as np
import cv_bridge
import rospy

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

"""
Door detector class. Creates a ros Image subscriber with a callback
function that uses cv_bridge to convert the image to a numpy array. Weights are
passed into the readNet function to allow the door to be detected. Detections
are then published as a Detection2D ROS publisher.
"""

CLASSES = ["door", "handle", "cabinet", "refrigerator"]
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
        self.detection_pub = rospy.Publisher(
            "/door_detections", Detection2D, queue_size=1
        )

        self.rotate_left = args.rotate_left
        self.rotate_right = args.rotate_right
        self.rotate_down = args.rotate_down

        self.raw_image = None
        self.timer_loop = rospy.Timer(rospy.Duration(0.5), self.timer_callback)

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

        net = cv2.dnn.readNet(
            "/home/cst/ws_spot/src/door_detector_ros/src/yolo-door.weights",
            "/home/cst/ws_spot/src/door_detector_ros/src/yolo-door.cfg",
        )

        scale = 0.00392
        blob = cv2.dnn.blobFromImage(
            image, scale, (416, 416), (0, 0, 0), True, crop=False
        )
        net.setInput(blob)
        outs = net.forward(self.get_output_layers(net))

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
        self.detection_pub.publish(
            self.get_detections(class_ids, confidences, boxes)
        )

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
            rospy.loginfo("Class: %s, Confidence: %s", CLASSES[class_ids[i]], confidences[i])
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

    def image_callback(self, data):
        try:
            self.raw_image = data
            self.frame_id = data.header.frame_id
        except cv_bridge.CvBridgeError as e:
            rospy.logerr(e)


if __name__ == "__main__":
    # argument parsing: create argparse parser, parse for rotation-indicator flag
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotate_left', '-l', action='store_true')
    parser.add_argument('--rotate_right', '-r', action='store_true')
    parser.add_argument('--rotate_down', '-d', action='store_true')

    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("door_detector", anonymous=True)
    DoorDetector(args)
    rospy.spin()
