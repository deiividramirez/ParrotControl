import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys
import os

if len(sys.argv) > 1:
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

PLOTTING3D(POINTS, label="3D points", color="royalblue")

##############################################################################################################
# Desired position for the camera
target_x, target_y, target_z = 0, 0.75, 3
target_roll, target_pitch, target_yaw = np.deg2rad([0, 0, 0])
target_pose = np.array(
    [[target_x], [target_y], [target_z], [target_roll], [target_pitch], [target_yaw]]
)
print(f"Target position: {target_x, target_y, target_z}")
print(f"Target orientation: {target_roll, target_pitch, target_yaw}")

# Camera parameters
DESIRED_CAMERA = PlanarCamera()
DESIRED_CAMERA.set_position(*target_pose[:, 0])
DESIRED_K = DESIRED_CAMERA.K
DESIRED_K_inv = np.linalg.inv(DESIRED_K)

DESIRED_DATA = Data()
DESIRED_DATA.feature = DESIRED_CAMERA.projection(POINTS.T, POINTS.shape[0])
DESIRED_DATA.inSphere, DESIRED_DATA.inNormalPlane = sendToSphere(
    DESIRED_DATA.feature, DESIRED_K_inv
)

PICTURE(DESIRED_DATA.feature, color="r", label="Desired position", z_order=2)
##############################################################################################################
# Initial position for the camera
initial_x, initial_y, initial_z = -1, 3, 8
initial_roll, initial_pitch, initial_yaw = np.deg2rad([0, 20, 0])
init_pose = np.array(
    [
        [initial_x],
        [initial_y],
        [initial_z],
        [initial_roll],
        [initial_pitch],
        [initial_yaw],
    ]
)
print(f"Initial position: {initial_x, initial_y, initial_z}")
print(f"Initial orientation: {initial_roll, initial_pitch, initial_yaw}")

# Camera parameters
ACTUAL_CAMERA = PlanarCamera()
ACTUAL_CAMERA.set_position(*init_pose[:, 0])
ACTUAL_K = ACTUAL_CAMERA.K
ACTUAL_K_inv = np.linalg.inv(ACTUAL_K)

ACTUAL_DATA = Data()
ACTUAL_DATA.feature = ACTUAL_CAMERA.projection(POINTS.T, POINTS.shape[0])
_, ACTUAL_DATA.inNormalPlane = sendToSphere(ACTUAL_DATA.feature, ACTUAL_K_inv)

ACTUAL_CAMERA.draw_camera(PLOTTING3D.ax, "k", scale=0.2, label="Initial position")
PICTURE(ACTUAL_DATA.feature, color="k", label="Initial position", z_order=2)
##############################################################################################################

# Time and simulation parameters
dt = 0.033
t0 = 0
tf = 15
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
errorArray = np.zeros((time.shape[0], 12))
errorPixArray = np.zeros((time.shape[0]))
controlInput = np.zeros((time.shape[0], 6))
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

    L = guoL(distancesArray, ACTUAL_DATA.inSphere, control=CONTROL)
    L_inv = np.linalg.pinv(L)

    # U = np.vstack((kp * L_inv @ error_control, kw * Lw_inv @ error_pix.reshape(-1, 1)))
    U = np.vstack((kp * L_inv @ error_control, np.zeros((3, 1))))

    if t % 120 == 0:
        print(
            f"\t[INFO] Calculating L matrix with GUO control law >> {'1/dist' if CONTROL == 1 else 'dist'}"
        )
        print(
            f"\nTime: {time[t]} - Iteration: {t} out of {time.shape[0]}",
            f"Input: {U[:, 0].reshape(-1, 1).T}",
            f"Pose: {actual_pose[:, 0].reshape(-1, 1).T}",
            f"Pose* {target_pose[:, 0].reshape(-1, 1).T}",
            f"Error: {error_norm}",
            f"Error pix: {np.linalg.norm(error_pix)}",
            sep="\n",
        )

    # Saving data
    controlInput[t] = U[:, 0]
    pose[t] = actual_pose[:, 0]
    imageTrajectory[t] = ACTUAL_DATA.feature
    errorArray[t, :6] = pose[t] - target_pose[:, 0]
    errorArray[t, 6:] = error_control.reshape(-1)
    errorPixArray[t] = np.linalg.norm(error_pix)

    # if np.linalg.norm(error_pix) < 0.001:
    #     END = t
    #     break

##############################################################################################################
# Plotting
DESIRED_CAMERA.set_position(*target_pose[:, 0])
DESIRED_CAMERA.draw_camera(PLOTTING3D.ax, "r", scale=0.2, label="Desired position")
ACTUAL_CAMERA.draw_camera(PLOTTING3D.ax, "b", scale=0.2, label="Actual position")

PLOTTING3D.trajectory(pose[:END], color="b")

PICTURE.trajectory(imageTrajectory[:END], color="b")

PLOTTING3D.fig.savefig(
    f"{MAIN_PATH}/out/Plotting3DLeader{'1_dist' if CONTROL == 1 else 'dist'}.svg",
    format="svg",
    transparent=True,
    bbox_inches="tight",
    pad_inches=0.1,
)
# PICTURE.fig.savefig(f"Plotting image {CONTROL}.svg", format="svg", transparent=True, bbox_inches="tight", pad_inches=0.1)

summaryPlot(
    time[:END],
    controlInput[:END],
    pose[:END],
    errorArray[:END],
    errorPixArray[:END],
    CONTROL=(0, CONTROL),
    PATH=MAIN_PATH,
)


##############################################################################################################
PICTURE.show()
PLOTTING3D.show()
