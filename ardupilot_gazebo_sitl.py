#!/usr/bin/env python3
import math
import time
import sys
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from dronekit import connect, VehicleMode, LocationGlobalRelative
from ultralytics import YOLO

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
connection_string = '127.0.0.1:14550'
yolo_model_path = "/home/keerthu/Development/drone_box_detection/best.pt"
video_resolution = [640, 480]

vehicle_airspeed = 1.0     # m/s
takeoff_altitude = 5.0     # meters
box_location_x = 2.0        # meters
box_location_y = 2.0        # meters
FOV_x = 66.0
FOV_y = 40.0

# -------------------------------------------------------------------
# DRONEKIT FUNCTIONS
# -------------------------------------------------------------------
def get_location_metres(original_location, dNorth, dEast, alt):
    earth_radius = 6378137.0
    dLat = dNorth / earth_radius
    dLon = dEast / (earth_radius * math.cos(math.radians(original_location.lat)))
    newlat = original_location.lat + math.degrees(dLat)
    newlon = original_location.lon + math.degrees(dLon)
    return LocationGlobalRelative(newlat, newlon, alt)

def arm_and_takeoff(vehicle, target_altitude):
    print("Basic pre-arm checks")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed and not vehicle.mode.name=='GUIDED':
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(target_altitude)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f" Altitude: {alt:.2f} m")
        if alt >= target_altitude * 0.90:
            print("Reached target altitude")
            break
        time.sleep(1)

def goto(vehicle, location_dx, location_dy, target_alt):
    current_location = vehicle.location.global_relative_frame
    target_location = get_location_metres(current_location, location_dx, location_dy, target_alt)
    vehicle.simple_goto(target_location)
    time.sleep(10)

# -------------------------------------------------------------------
# ROS 2 NODE CLASS
# -------------------------------------------------------------------
class CameraYOLONode(Node):
    def __init__(self, vehicle, model):
        super().__init__('camera_yolo_node')
        self.vehicle = vehicle
        self.model = model
        self.bridge = CvBridge()

        # Subscribe to Gazebo camera feed
        self.subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10)
        self.subscription  # prevent unused warning
        self.get_logger().info("Subscribed to /camera")

        self.FOV_x = FOV_x
        self.FOV_y = FOV_y

    def image_callback(self, msg):
        try:
            self.get_logger().info("Callback triggered")
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        results = self.model.predict(source=frame, verbose=False)
        image_height, image_width = frame.shape[:2]

        found_box = False
        box_conf = 0.0
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            x, y, w, h = boxes.xywh[0]
            box_conf = float(boxes.conf[0])
            x_center, y_center = float(x), float(y)
            found_box = True
            break

        if found_box and box_conf > 0.3:
            dx_pixels = x_center - (image_width / 2)
            dy_pixels = y_center - (image_height / 2)
            theta_x = dx_pixels / (image_width / 2) * (self.FOV_x / 2)
            theta_y = dy_pixels / (image_height / 2) * (self.FOV_y / 2)
            offset_x = h * math.tan(math.radians(theta_x))
            offset_y = h * math.tan(math.radians(theta_y))

            yaw_deg = self.vehicle.heading if self.vehicle.heading is not None else 0.0
            yaw_rad = math.radians(yaw_deg)
            dNorth = offset_y * math.cos(yaw_rad) - offset_x * math.sin(yaw_rad)
            dEast = offset_y * math.sin(yaw_rad) + offset_x * math.cos(yaw_rad)

            self.get_logger().info(f"Target detected at offset: N={dNorth:.2f}m, E={dEast:.2f}m (conf={box_conf:.2f})")

            # Send goto command
            goto(dNorth, dEast, 2)
            self.vehicle.mode = VehicleMode("LAND")
        else:
            self.get_logger().info("Target not detected, Return to launch")
            self.vehicle.mode = VehicleMode("RTL")

        # Display camera
        cv2.imshow("Gazebo Camera", frame)
        cv2.waitKey(1)

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    print(f"Connecting to vehicle on: {connection_string}")
    vehicle = connect(connection_string, baud=57600, wait_ready=True)
    vehicle.airspeed = vehicle_airspeed

    # Load the YOLO model
    print("Loading YOLO model...")
    model = YOLO(yolo_model_path)

    # Takeoff and navigate before starting detection
    arm_and_takeoff(vehicle, takeoff_altitude)
    goto(vehicle, box_location_x, box_location_y, takeoff_altitude)

    # Start ROS 2 camera-YOLO node
    node = CameraYOLONode(vehicle, model)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt, exiting...")
    finally:
        print("Cleaning up resources...")
        node.destroy_node()
        if vehicle is not None:
            vehicle.mode = VehicleMode("SMART_RTL")
            time.sleep(1)
            vehicle.armed = False
            time.sleep(1)
        vehicle.close()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
