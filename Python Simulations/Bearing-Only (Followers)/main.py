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


POINTS = np.loadtxt(f"{MAIN_PATH}/points.txt")
PLOTTING3D = Plotting3D()
PICTURE = PlottingImage((0, 640), (0, 480))

PLOTTING3D(POINTS)

##############################################################################################################
# Desired position for the camera
target_x, target_y, target_z = 0, 0, 3
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
DESIRED_DATA.bearings = middlePoint(DESIRED_DATA.inSphere)

DESIRED_CAMERA.draw_camera(PLOTTING3D.ax, "r", scale=0.2)
PICTURE(DESIRED_DATA.feature, color="r", label="Desired position", z_order=2)
##############################################################################################################
# Initial position for the camera
initial_x, initial_y, initial_z = 0, 1, 5
initial_roll, initial_pitch, initial_yaw = np.deg2rad([0, 30, 0])
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

ACTUAL_CAMERA.draw_camera(PLOTTING3D.ax, "k", scale=0.2)
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
linear_integral = np.zeros((6, 1))
U = np.zeros((6, 1))

lastR = None

# Error
error = np.zeros((6, 1))
error_norm = 0
error_control = np.zeros((6, 1))
error_pose = np.zeros((6, 1))

# Array for saving data
errorArray = np.zeros((time.shape[0], 6))
errorPixArray = np.zeros((time.shape[0]))
controlInput = np.zeros((time.shape[0], 6))
pose = np.zeros((time.shape[0], 6))
integralArray = np.zeros((time.shape[0], 6))
imageTrajectory = np.zeros((time.shape[0], POINTS.shape[0], 2))

# Auxiliary variables
I3 = np.eye(3)
kp = 1.3
kw = 0.1
ki = 0.02

KV = np.eye(6)
KV[:3, :3] = kp * I3
KV[3:, 3:] = kw * I3

KI = ki * np.eye(6)
KI[:3, :3] = ki * I3
KI[3:, 3:] = 0 * I3

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

toplot = [[[], [], []], [[], [], []]]
##############################################################################################################
for t in range(time.shape[0]):
    actual_pose += dt * U
    ACTUAL_CAMERA.set_position(*actual_pose[:, 0])

    ACTUAL_DATA.feature = ACTUAL_CAMERA.projection(POINTS.T, POINTS.shape[0])
    ACTUAL_DATA.inSphere, ACTUAL_DATA.inNormalPlane = sendToSphere(
        ACTUAL_DATA.feature, ACTUAL_K_inv
    )
    # ACTUAL_DATA.bearings = sendToSphere(middlePoint(ACTUAL_DATA.feature), ACTUAL_K_inv)
    ACTUAL_DATA.bearings = middlePoint(ACTUAL_DATA.inSphere)

    # print(f"In sphere: {ACTUAL_DATA.inSphere}")
    # print(f"In sphere: {DESIRED_DATA.inSphere}")

    # print(f"Actual bearings: {ACTUAL_DATA.bearings}")
    # print(f"Desired bearings: {DESIRED_DATA.bearings}")

    error_control = bearingControl(
        ACTUAL_DATA, DESIRED_DATA, ACTUAL_K, ACTUAL_K_inv, lastR, CONTROL, toplot
    )
    error_norm = np.linalg.norm(error_control)
    error_pix = ACTUAL_DATA.inNormalPlane - DESIRED_DATA.inNormalPlane

    linear_integral += dt * np.sign(error_control[:, 0]).reshape(-1, 1)
    for i in range(linear_integral.shape[0]):
        if abs(error_control[i, 0]) <= 0.025:
            linear_integral[i, 0] = 0
    U = (
        KV @ np.abs(error_control) ** (0.5) * np.sign(error_control)
        + KI @ linear_integral
    )
    # U[1] *= 2
    # U[4] *= 1.2
    # U[0, 0] = np.cos(2 * time[t]) * 2
    # U[1, 0] = np.sin(2 * time[t]) * 2
    # U[2, 0] = np.sin(2 * time[t]) * 2
    # U[3, 0] = np.sin(time[t]) / 2
    # U[4, 0] = np.cos(time[t]) / 2
    # U[5, 0] = -np.sin(time[t]) / 2
    if t % 120 == 0:
        print(
            f"\nTime: {time[t]} - Iteration: {t} out of {time.shape[0]}",
            f"Input: {U[:, 0].reshape(-1, 1).T}",
            f"Integral: {(KI @ linear_integral[:, 0]).reshape(-1, 1).T}",
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
    errorArray[t] = error_control[:, 0]
    errorPixArray[t] = np.linalg.norm(error_pix)
    integralArray[t] = KI @ linear_integral[:, 0]

    if np.linalg.norm(error_pix) < 0.001:
        END = t
        break

##############################################################################################################
# Plotting
ACTUAL_CAMERA.draw_camera(PLOTTING3D.ax, "b", scale=0.2)
PLOTTING3D.trajectory(pose[:END], color="b", label="Actual position")

PICTURE.trajectory(imageTrajectory[:END], color="b")

# fig, ax = plt.subplots(4, 1, sharex=True)
# ax = ax.flatten()
# for i in range(0, time.shape[0], 10):
#     [i.cla() for i in ax]
    # animation(
    #     time[: i + 1],
    #     controlInput[: i + 1],
    #     pose[: i + 1],
    #     errorArray[: i + 1],
    #     errorPixArray[: i + 1],
    #     integralArray[: i + 1],
    #     fig,
    #     ax,
    # )
    # plt.pause(0.001)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.autoscale(enable=True, axis="both", tight=True)
for i in range(0, time.shape[0]-1, 2):
    ax.cla()
    animation3D(DESIRED_CAMERA, ACTUAL_CAMERA, pose[i], fig, ax)
    plt.pause(0.001)
# summaryPlot(
#     time[:END],
#     controlInput[:END],
#     pose[:END],
#     errorArray[:END],
#     errorPixArray[:END],
#     integralArray[:END],
# )

fig, ax = plt.subplots(2, 1, sharex=True)
for index in range(DESIRED_DATA.bearings.shape[0]):
    toplot[index][0] = np.array(toplot[index][0])
    toplot[index][1] = np.array(toplot[index][1])
    toplot[index][2] = np.array(toplot[index][2])
    ax[index].plot(time[: END + 1], toplot[index][0], label=f"Pitch {index}")
    ax[index].plot(time[: END + 1], toplot[index][1], label=f"Yaw {index}")
    ax[index].plot(time[: END + 1], toplot[index][2], label=f"Roll {index}")
    ax[index].legend(loc="best")


##############################################################################################################
PICTURE.show()
PLOTTING3D.show()
