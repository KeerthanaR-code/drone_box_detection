# ArduPilot SITL + Gazebo + DroneKit + YOLOv8 Visual Guidance

This project demonstrates a **vision-based object detection and navigation system** for a drone using **ArduPilot**, **Gazebo**, and **DroneKit**.  
The drone uses a **YOLOv8** model and an onboard camera feed to detect a target box on the ground and autonomously fly and land on it.

# Libraries Used

| Component              | Description                                      |
| ---------------------- | ------------------------------------------------ |
| **ArduPilot**          | Flight control firmware                          |
| **Gazebo Harmonic**    | Physics and 3D visualization simulator           |
| **DroneKit-Python**    | Python API to control ArduPilot via MAVLink      |
| **Ultralytics YOLOv8** | Object detection framework                       |
| **OpenCV**             | Video capture and image processing               |
| **Ubuntu 22.04**       | Video capture and image processing               |
| **Python 3.8+**        | Core runtime environment                         |

# Prerequisities Installation

## 1️⃣ Install ArduPilot and SITL
```bash
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile
```

## 2️⃣ Install Gazebo and ArduPilot plugin
```bash
sudo apt install gazebo libgazebo-dev
git clone https://github.com/ArduPilot/ardupilot_gazebo.git
cd ardupilot_gazebo
mkdir build && cd build
cmake ..
make -j4
sudo make install
```

## 3️⃣ Python dependencies
```bash
pip install dronekit pymavlink opencv-python ultralytics numpy
```

# Project Structure
```bash
📦 Ardupilot_VisualGuidance
├── Ardupilot_Gazebo_SITL.py       # Main control script
├── README.md                      # Project documentation
├── data.yaml                      # YOLO configuration file
├── models/
│   └── box_target/                # Gazebo model of the target box
│       ├── model.config
│       └── model.sdf
├── runs/
│   └── detect/
│       └── train/
│            └── weights/
│               └── best.pt       # YOLOv8 trained weights
│       └── val  

```

# Running a simulation

## 1️⃣ Start ArduPilot SITL
```bash
sim_vehicle.py -v ArduCopter -f gazebo-iris --console --map
```

## 2️⃣ Launch Gazebo with box model
```bash
gz sim ~/ardupilot_gazebo/worlds/iris_arducopter_runway.sdf
```

## 3️⃣ Run the Python script
```bash
python3 Ardupilot_Gazebo_SITL.py
```

# How It Works

The drone takes off to a defined altitude and flies to a pre-defined waypoint.

YOLOv8 runs inference on each frame from the camera feed.

When the box is detected,

The pixel offset from the image center is converted to real-world meters.

The drone flies toward the box using position or velocity control.

If detection is lost,

The drone hovers or performs a slow search yaw.

After a configurable timeout, it returns to launch (RTL).

# Author

Keerthana Radhakrishnan

LinkedIn: www.linkedin.com/in/keerthanaradhakrishnan

