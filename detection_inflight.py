#!/usr/bin/env python3
import sys
import time
import math
import numpy as np
import cv2
from ultralytics import YOLO
from dronekit import connect, VehicleMode, LocationGlobalRelative

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
connection_string = '127.0.0.1:14550'
video_source = 0        
video_resolution = [640,480]

yolo_model_path = "../runs/detect/train/weights/best.pt"

vehicle_airspeed = 1.0    # m/s
takeoff_altitude = 5.0    # meters
box_location_x = 5.0      # meters, payload location in SITL
box_location_y = 5.0      # meters, payload location in SITL
search_timeout = 5.0
lost_timeout = 30.0
FOV_x = 66.0
FOV_y = 40.0
found_box = False
box_conf = 0.0

# -------------------------------------------------------------------
# DRONEKIT FUNCTIONS
# -------------------------------------------------------------------
#Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the specified `original_location`
def get_location_metres(original_location, dNorth, dEast, alt):
    if alt is None:
        alt = original_location.alt
    earth_radius = 6378137.0  # Radius of "spherical" earth
    # Coordinate offsets in radians
    dLat = dNorth / earth_radius
    dLon = dEast / (earth_radius * cos(radians(original_location.lat)))
    newlat = original_location.lat + degrees(dLat)
    newlon = original_location.lon + degrees(dLon)
    return LocationGlobalRelative(newlat, newlon, alt )

# Arms vehicle and flies to takeoff_altitude.
def arm_and_takeoff(takeoff_altitude):
    print("Basic pre-arm checks")
    while not vehicle.is_armable:
        print("Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    # Arming vehicle in GUIDED mode
    vehicle.mode    = VehicleMode("GUIDED")
    vehicle.armed   = True
    # Confirm vehicle armed before attempting to take off
    while not vehicle.armed and not vehicle.mode.name=='GUIDED':
        print("Getting ready to take off ...")
        time.sleep(1)

    # Take off to target altitude
    print(f"Taking off!")
    vehicle.simple_takeoff(takeoff_altitude) 

    # Wait until the vehicle reaches a safe height before processing the goto
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"Altitude: {alt:.2f} m")
        #Break and return from function just below target altitude.
        if alt >= takeoff_altitude*0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

# Create goto commands for waypoints (Point A: 0, 0, takeoff_altitude; Point B: 0, 5, takeoff_altitude)
def goto(location_dx, location_dy, target_alt):
    current_location = vehicle.location.global_relative_frame
    target_location = get_location_metres(current_location, location_dx, location_dy, target_alt)
    vehicle.simple_goto(target_location)   
    # sleep so we can see the change in map
    time.sleep(10)

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    # Capture video from camera
    cap = cv2.VideoCapture(video_source)                 # sourcing video from camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_resolution[1])
    if not cap.isOpened():
        print("Error opening camera")
        cap.release()
        sys.exit(0)

    # Load the YOLO model
    print("Loading YOLO model...")
    model = YOLO(yolo_model_path)  # load a custom model

    # Connect to the Vehicle
    print(f'Connecting to vehicle on: {connection_string}')
    vehicle = connect(connection_string, baud=57600, wait_ready=True)
    vehicle.airspeed = vehicle_airspeed

    try:
        # Takeoff and navigate before starting detection
        arm_and_takeoff(takeoff_altitude)
        time.sleep(1)
        goto(box_location_x, box_location_y, takeoff_altitude)

        print("Starting YOLO-based guidance")
        while True:
            # Get frames from video feed
            has_frame, frame = cap.read()
            if not has_frame:
                print("Error opening frames")
                continue

            # Predict with the model
            results = model(frame, stream=True)

            # Display results
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                x, y, w, h = boxes.xywh[0]
                #names = [result.names[cls.item()] for cls in boxes.cls.int()]
                box_conf = boxes.conf
                x_center, y_center = float(x), float(y)
                found_box = True
                break

            image_height, image_width = result.orig_shape[0], result.orig_shape[1]
            if found_box and box_conf > 0.3:
                # Calculate box location on ground w.r.t to drone pose
                dx_pixels = x_center - (image_width/2)
                dy_pixels = y_center - (image_height/2)
                theta_x = dx_pixels / (image_width / 2) * (FOV_x / 2)
                theta_y = dy_pixels / (image_height / 2) * (FOV_y / 2)
                offset_x = h * tan(theta_x)
                offset_y = h * tan(theta_y)
                yaw_deg = vehicle.heading if vehicle.heading is not None else 0.0
                yaw_rad = math.radians(yaw_deg)
                dNorth =  offset_y * math.cos(yaw_rad) - offset_x * math.sin(yaw_rad)
                dEast  =  offset_y* math.sin(yaw_rad) + offset_x * math.cos(yaw_rad)
                # Command drone to go to box location at an altitude of 2 meters
                goto(dNorth, dEast, 2)
                vehicle.mode = VehicleMode("LAND")

    except Exception as e:
        print("Exception in main loop:", e)

    finally:
        print("Cleaning up resources...")
        if vehicle is not None:
            print("Setting vehicle to LOITER and disarming (if armed).")
            vehicle.mode = VehicleMode("LOITER")
            time.sleep(1)
            if vehicle.armed:
                vehicle.armed = False
            time.sleep(1)
            vehicle.close()
        if cap is not None:
            cap.release()

    print("Exited.")

if __name__ == '__main__':
    main()