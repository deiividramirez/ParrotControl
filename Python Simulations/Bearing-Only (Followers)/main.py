import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys
import os

if (len(sys.argv) > 1):
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
PLOTTING3D = Plotting3D(f"Control {'Orthogonal projection' if CONTROL == 1 else 'Difference of bearings'}")
PICTURE = PlottingImage((0, 640), (0, 480))

PLOTTING3D(POINTS, label="3D points", color="k")

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
DESIRED_DATA.bearings = middlePoint(DESIRED_DATA.inSphere)

# DESIRED_CAMERA.draw_camera(PLOTTING3D.ax, "r", scale=0.2, label="Desired position")
PICTURE(DESIRED_DATA.feature, color="r", label="Desired position", z_order=2)
##############################################################################################################
# Initial position for the camera
initial_x, initial_y, initial_z = -1, 3, 8
initial_roll, initial_pitch, initial_yaw = np.deg2rad([0, 0, 0])
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
tf = 60
time = np.linspace(t0, tf, int((tf - t0) / dt) + 1, endpoint=True)
END = time.shape[0] - 1

# Control parameters
linear_velocity = np.zeros((3, 1))
angular_velocity = np.zeros((3, 1))
linear_integral = np.zeros((6, 1))
U = np.zeros((6, 1))

# Error
error = np.zeros((6, 1))
error_norm = 0
error_control = np.zeros((6, 1))
error_pose = np.zeros((6, 1))

# Array for saving data
errorArray = np.zeros((time.shape[0], 9))
errorPixArray = np.zeros((time.shape[0]))
controlInput = np.zeros((time.shape[0], 6))
pose = np.zeros((time.shape[0], 6))
integralArray = np.zeros((time.shape[0], 6))
imageTrajectory = np.zeros((time.shape[0], POINTS.shape[0], 2))
pointsTrajectory = np.zeros((time.shape[0], POINTS.shape[0], 3))

# Auxiliary variables
I3 = np.eye(3)
kp = 2.3
kw = 0.1
ki = 0.05
# ki = 0.

KV = np.eye(6)
KV[:3, :3] = kp * I3
KV[3:, 3:] = kw * I3

KI = ki * np.eye(6)
# KI[:3, :3] = ki * I3
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

# fx = lambda x: np.sin(x / 2) / 2
# fy = lambda x: np.cos(x / 2) / 2
# fz = lambda x: 0
fx = lambda x: x / 20
fy = lambda x: np.cos(x / 2) / 4
fz = lambda x: np.sin(x / 2) / 4
fx = lambda x: 0
fy = lambda x: 0
fz = lambda x: 0
##############################################################################################################
for t in range(time.shape[0]):
    actual_pose += dt * U
    ACTUAL_CAMERA.set_position(*actual_pose[:, 0])

    POINTS += dt * pointsMove(time[t], fx, fy, fz, POINTS.shape[0])
    target_pose[0, 0] += dt * fx(time[t])
    target_pose[1, 0] += dt * fy(time[t])
    target_pose[2, 0] += dt * fz(time[t])

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
        ACTUAL_DATA, DESIRED_DATA, ACTUAL_K, ACTUAL_K_inv, CONTROL, toplot
    )
    error_norm = np.linalg.norm(error_control)
    error_pix = ACTUAL_DATA.inNormalPlane - DESIRED_DATA.inNormalPlane

    linear_integral += dt * np.sign(error_control)
    # for i in range(linear_integral.shape[0]):
    #     if abs(error_control[i, 0]) <= 0.025:
    #         linear_integral[i, 0] = 0
    U = KV @ error_control
    # U = (
    #     KV @ np.abs(error_control) ** (0.5) * np.sign(error_control)
    #     + KI @ linear_integral
    # )

    U[2] *= 2
    
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
    pointsTrajectory[t] = POINTS
    # errorArray[t] = np.abs(error_control[:, 0])
    errorBearing = np.linalg.norm(ACTUAL_DATA.bearings - DESIRED_DATA.bearings, axis=0)
    errorToSend = np.concatenate((pose[t] - target_pose[:, 0], errorBearing))
    errorArray[t] = errorToSend
    # errorArray[t] = pose[t] - target_pose[:, 0]
    errorPixArray[t] = np.linalg.norm(error_pix)
    integralArray[t] = KI @ linear_integral[:, 0]

    # if np.linalg.norm(error_pix) < 0.001:
    #     END = t
    #     break

##############################################################################################################
# Plotting
DESIRED_CAMERA.set_position(*target_pose[:, 0])
DESIRED_CAMERA.draw_camera(PLOTTING3D.ax, "r", scale=0.2, label="Desired position")
ACTUAL_CAMERA.draw_camera(PLOTTING3D.ax, "b", scale=0.2, label="Actual position")

PLOTTING3D.trajectory(pose[:END], color="b")
PLOTTING3D.trajectoryPoints(pointsTrajectory[:END], color="k", alpha=0.5)
PLOTTING3D(POINTS, label="3D points", color="royalblue")

PICTURE.trajectory(imageTrajectory[:END], color="b")

PLOTTING3D.fig.savefig(f"{MAIN_PATH}/out/Plotting3DFollower{'OrthogonalProj' if CONTROL == 1 else 'Difference'}.svg", format="svg", transparent=True, bbox_inches="tight", pad_inches=0.1)
# PICTURE.fig.savefig(f"Plotting image {CONTROL}.svg", format="svg", transparent=True, bbox_inches="tight", pad_inches=0.1)

summaryPlot(
    time[:END],
    controlInput[:END],
    pose[:END],
    errorArray[:END],
    errorPixArray[:END],
    integralArray[:END],
    CONTROL=(1,CONTROL),
    PATH=MAIN_PATH,
)

# fig, ax = plt.subplots(2, 1, sharex=True)
# for index in range(DESIRED_DATA.bearings.shape[0]):
#     toplot[index][0] = np.array(toplot[index][0])
#     toplot[index][1] = np.array(toplot[index][1])
#     toplot[index][2] = np.array(toplot[index][2])
#     ax[index].plot(time[: END + 1], toplot[index][0], label=f"Pitch {index}")
#     ax[index].plot(time[: END + 1], toplot[index][1], label=f"Yaw {index}")
#     ax[index].plot(time[: END + 1], toplot[index][2], label=f"Roll {index}")
#     ax[index].legend(loc="best")


##############################################################################################################
PICTURE.show()
PLOTTING3D.show()
