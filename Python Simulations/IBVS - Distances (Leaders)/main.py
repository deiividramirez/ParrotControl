import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys
import os

if (len(sys.argv) > 1) and (sys.argv[1] == "CONTROL"):
    CONTROL = sys.argv[1]
else:
    CONTROL = 1

ACTUAL_PATH = pathlib.Path(__file__).parent.absolute()
MAIN_PATH = ACTUAL_PATH.parent.absolute()
sys.path.append(f"{MAIN_PATH}")


from CameraModels.PlanarCamera import PlanarCamera
from Utils.ImageJacobian import ImageJacobian
from Funcs import *

tempPoints = np.loadtxt(f"{MAIN_PATH}/points.txt")
POINTS = np.array([tempPoints[2], tempPoints[3], tempPoints[7], tempPoints[5]])
# POINTS = np.loadtxt(f"{MAIN_PATH}/points.txt")
PLOTTING3D = Plotting3D()
PICTURE = PlottingImage((0, 640), (0, 480))

PLOTTING3D(POINTS)

##############################################################################################################
# Desired position for the camera
target_x, target_y, target_z = 0, 0, 5
target_roll, target_pitch, target_yaw = np.deg2rad([0, 0, 0])
print(f"Target position: {target_x, target_y, target_z}")
print(f"Target orientation: {target_roll, target_pitch, target_yaw}")

# Camera parameters
DESIRED_CAMERA = PlanarCamera()
DESIRED_CAMERA.set_position(
    target_x, target_y, target_z, target_roll, target_pitch, target_yaw
)
DESIRED_CAMERA.draw_camera(PLOTTING3D.ax, "b", scale=0.2)
DESIRED_K = DESIRED_CAMERA.K
DESIRED_K_inv = np.linalg.inv(DESIRED_K)

DESIRED_DATA = Data()
DESIRED_DATA.feature = DESIRED_CAMERA.projection(POINTS.T, POINTS.shape[0])
DESIRED_DATA.inSphere, DESIRED_DATA.inNormalPlane = sendToSphere(
    DESIRED_DATA.feature, DESIRED_K_inv
)

PICTURE(DESIRED_DATA.feature, color="b", label="Desired position", z_order=2)
##############################################################################################################
# Initial position for the camera
initial_x, initial_y, initial_z = 0, 1, 10
initial_roll, initial_pitch, initial_yaw = np.deg2rad([0, 0, 0])
print(f"Initial position: {initial_x, initial_y, initial_z}")
print(f"Initial orientation: {initial_roll, initial_pitch, initial_yaw}")

# Camera parameters
ACTUAL_CAMERA = PlanarCamera()
ACTUAL_CAMERA.set_position(
    initial_x, initial_y, initial_z, initial_roll, initial_pitch, initial_yaw
)
ACTUAL_CAMERA.draw_camera(PLOTTING3D.ax, "k", scale=0.2)
ACTUAL_K = ACTUAL_CAMERA.K
ACTUAL_K_inv = np.linalg.inv(ACTUAL_K)

ACTUAL_DATA = Data()
ACTUAL_DATA.feature = ACTUAL_CAMERA.projection(POINTS.T, POINTS.shape[0])
_, ACTUAL_DATA.inNormalPlane = sendToSphere(ACTUAL_DATA.feature, ACTUAL_K_inv)

PICTURE(ACTUAL_DATA.feature, color="k", label="Initial position", z_order=2)
##############################################################################################################

# Time and simulation parameters
dt = 0.033
t0 = 0
tf = 60
time = np.linspace(t0, tf, int((tf - t0) / dt) + 1, endpoint=True)
END = time.shape[0] - 1

# Control parameters
linear_velocity = np.zeros((3, 1))
angular_velocity = np.zeros((3, 1))
U = np.zeros((6, 1))

# Error
error = np.zeros((6, 1))
error_norm = 0
error_control = np.zeros((6, 1))
error_pose = np.zeros((6, 1))

# Array for saving data
controlInput = np.zeros((6, time.shape[0]))
pose = np.zeros((time.shape[0], 6))
imageTrajectory = np.zeros((time.shape[0], POINTS.shape[0], 2))

# Auxiliary variables
I3 = np.eye(3)
kp = 3
kw = 0.5

Lw = ImageJacobian(DESIRED_DATA.inNormalPlane.T, np.ones((POINTS.shape[0], 1)), 1)[
    :, 3:
]
Lw_inv = np.linalg.pinv(Lw)

actual_pose = np.array(
    [
        [initial_x],
        [initial_y],
        [initial_z],
        [initial_roll],
        [initial_pitch],
        [initial_yaw],
    ]
)

##############################################################################################################
for t in range(time.shape[0]):
    print(f"\nTime: {time[t]}")

    actual_pose += dt * U
    ACTUAL_CAMERA.set_position(*actual_pose[:, 0])

    ACTUAL_DATA.feature = ACTUAL_CAMERA.projection(POINTS.T, POINTS.shape[0])
    ACTUAL_DATA.inSphere, ACTUAL_DATA.inNormalPlane = sendToSphere(
        ACTUAL_DATA.feature, ACTUAL_K_inv
    )
    distancesArray = distancesInSphere(
        ACTUAL_DATA.inSphere, DESIRED_DATA.inSphere, CONTROL
    )

    error_control = np.array([i.dist - i.dist2 for i in distancesArray]).reshape(-1, 1)
    error_norm = np.linalg.norm(error_control)
    error_pix = ACTUAL_DATA.inNormalPlane - DESIRED_DATA.inNormalPlane

    L = guoL(distancesArray, ACTUAL_DATA.inSphere, CONTROL)
    L_inv = np.linalg.pinv(L)

    U = np.vstack((kp * L_inv @ error_control, kw * Lw_inv @ error_pix.reshape(-1, 1)))
    print(
        f"Input: {U[:, 0].reshape(-1, 1).T}"
        f"\nPose: {actual_pose[:, 0].reshape(-1, 1).T}"
        f"\nError: {error_norm}"
    )
    # Saving data
    controlInput[:, t] = U[:, 0]
    pose[t, :] = actual_pose[:, 0]
    imageTrajectory[t] = ACTUAL_DATA.feature

    if np.linalg.norm(error_pix) < 0.01:
        END = t
        break

##############################################################################################################
# Plotting
PICTURE.trajectory(imageTrajectory[:END], color="r")
PLOTTING3D.trajectory(pose[:END], color="r", label="Actual position")


##############################################################################################################
PICTURE.show()
PLOTTING3D.show()
