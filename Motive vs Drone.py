import matplotlib.pylab as plt
import numpy as np


def rotMat(roll, pitch, yaw) -> np.ndarray:
    R = np.array(
        [
            [
                np.cos(yaw) * np.cos(pitch),
                np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
                np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll),
            ],
            [
                np.sin(yaw) * np.cos(pitch),
                np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
                np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll),
            ],
            [
                -np.sin(pitch),
                np.cos(pitch) * np.sin(roll),
                np.cos(pitch) * np.cos(roll),
            ],
        ]
    )
    return R


def skewMat(v) -> np.ndarray:
    S = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return S


def normalize(v) -> np.ndarray:
    return v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v


marker_1 = np.array([1434.6, 803.3, 614.3]) / 1000
marker_2 = np.array([130.3, 689.2, 1470.7]) / 1000
drone = np.array([-669.28, 824.19, -1114.8]) / 1000

bearing_1_drone = np.array(
    [0.3239050805568695, -0.22209714353084564, 0.919651210308075]
)
bearing_2_drone = np.array(
    [-0.31427812576293945, -0.2254883348941803, 0.9221628308296204]
)

t = np.array([0.09, 0, 0])
eRc = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
eTc = np.vstack((np.hstack((eRc, t.reshape(3, 1))), np.array([0, 0, 0, 1])))

roll_1 = np.pi / 2
pitch_1 = 0
yaw_1 = np.pi

roll_2 = -np.pi / 2
pitch_2 = 0
yaw_2 = 0

roll_3 = np.deg2rad(0)
pitch_3 = np.deg2rad(90 - 35)
yaw_3 = np.deg2rad(0)

R_1 = rotMat(roll_1, pitch_1, yaw_1)
R_2 = rotMat(roll_2, pitch_2, yaw_2) @ rotMat(roll_3, pitch_3, yaw_3) @ R_1

bearing_1_drone = normalize(R_2.T @ (eTc @ np.hstack((bearing_1_drone, 1)))[0:3])
bearing_2_drone = normalize(R_2.T @ (eTc @ np.hstack((bearing_2_drone, 1)))[0:3])

print("R_1: ", R_1.round(3))
print("R_2: ", R_2.round(3))

print("\nmarker 1: ", marker_1)
print("marker 2: ", marker_2)
print("drone: ", drone)

marker_1_world = R_1.T @ marker_1
marker_2_world = R_1.T @ marker_2
drone_world = R_1.T @ drone

bearing_1_world = normalize(marker_2_world - drone_world)
bearing_2_world = normalize(marker_1_world - drone_world)

print("\nBearing 1 world: ", bearing_1_world, np.linalg.norm(bearing_1_world))
print("Bearing 1 drone: ", bearing_1_drone, np.linalg.norm(bearing_1_drone))

print("\nBearing 2 world: ", bearing_2_world, np.linalg.norm(bearing_2_world))
print("Bearing 2 drone: ", bearing_2_drone, np.linalg.norm(bearing_2_drone))

print("\nDifferece 1: ", np.linalg.norm(bearing_1_world - bearing_1_drone))
print("Differece 2: ", np.linalg.norm(bearing_2_world - bearing_2_drone))

# PLOTEO DE LOS EJES
ax = plt.figure().add_subplot(projection="3d", title="World Motive")
ejes_1 = np.vstack((R_1 @ np.eye(3), np.ones(3)))
ax.quiver(
    [0],
    [0],
    [0],
    [ejes_1[0, 0]],
    [ejes_1[1, 0]],
    [ejes_1[2, 0]],
    color="g",
    label="eje x",
    normalize=True,
)
ax.quiver(
    [0],
    [0],
    [0],
    [ejes_1[0, 1]],
    [ejes_1[1, 1]],
    [ejes_1[2, 1]],
    color="r",
    label="eje y",
    normalize=True,
)
ax.quiver(
    [0],
    [0],
    [0],
    [ejes_1[0, 2]],
    [ejes_1[1, 2]],
    [ejes_1[2, 2]],
    color="b",
    label="eje z",
    normalize=True,
)
ax.scatter([0], [0], [0], color="k", marker="o")
ax.scatter(
    [marker_1_world[0]], [marker_1_world[1]], [marker_1_world[2]], color="c", marker="o"
)
ax.scatter(
    [marker_2_world[0]], [marker_2_world[1]], [marker_2_world[2]], color="c", marker="o"
)
ax.scatter([drone_world[0]], [drone_world[1]], [drone_world[2]], color="m", marker="o")

ax.scatter(
    [drone_world[0] + bearing_1_world[0]],
    [drone_world[1] + bearing_1_world[1]],
    [drone_world[2] + bearing_1_world[2]],
    color="r",
    marker="o",
)
ax.scatter(
    [drone_world[0] + bearing_2_world[0]],
    [drone_world[1] + bearing_2_world[1]],
    [drone_world[2] + bearing_2_world[2]],
    color="r",
    marker="o",
)
#draw line from drone to 3*bearing
ax.plot(
    [drone_world[0], drone_world[0] + 3 * bearing_1_world[0]],
    [drone_world[1], drone_world[1] + 3 * bearing_1_world[1]],
    [drone_world[2], drone_world[2] + 3 * bearing_1_world[2]],
    color="r",
)
ax.plot(
    [drone_world[0], drone_world[0] + 3 * bearing_2_world[0]],
    [drone_world[1], drone_world[1] + 3 * bearing_2_world[1]],
    [drone_world[2], drone_world[2] + 3 * bearing_2_world[2]],
    color="r",
)

ax1 = plt.figure().add_subplot(projection="3d", title="Drone")
ejes_2 = np.vstack((R_2 @ np.eye(3), np.ones(3)))
ax1.quiver(
    [drone_world[0]],
    [drone_world[1]],
    [drone_world[2]],
    [ejes_2[0, 0]],
    [ejes_2[1, 0]],
    [ejes_2[2, 0]],
    color="g",
    label="eje x",
    normalize=True,
)
ax1.quiver(
    [drone_world[0]],
    [drone_world[1]],
    [drone_world[2]],
    [ejes_2[0, 1]],
    [ejes_2[1, 1]],
    [ejes_2[2, 1]],
    color="r",
    label="eje y",
    normalize=True,
)
ax1.quiver(
    [drone_world[0]],
    [drone_world[1]],
    [drone_world[2]],
    [ejes_2[0, 2]],
    [ejes_2[1, 2]],
    [ejes_2[2, 2]],
    color="b",
    label="eje z",
    normalize=True,
)
ax1.scatter([drone_world[0]], [drone_world[1]], [drone_world[2]], color="k", marker="o")
ax1.scatter(
    [marker_1_world[0]], [marker_1_world[1]], [marker_1_world[2]], color="c", marker="o"
)
ax1.scatter(
    [marker_2_world[0]], [marker_2_world[1]], [marker_2_world[2]], color="c", marker="o"
)
ax1.scatter([drone_world[0]], [drone_world[1]], [drone_world[2]], color="m", marker="o")

ax1.scatter(
    [drone_world[0] + bearing_1_drone[0]],
    [drone_world[1] + bearing_1_drone[1]],
    [drone_world[2] + bearing_1_drone[2]],
    color="r",
    marker="o",
)
ax1.scatter(
    [drone_world[0] + bearing_2_drone[0]],
    [drone_world[1] + bearing_2_drone[1]],
    [drone_world[2] + bearing_2_drone[2]],
    color="r",
    marker="o",
)
# draw line from drone to 3*bearing
ax1.plot(
    [drone_world[0], drone_world[0] + 3 * bearing_1_drone[0]],
    [drone_world[1], drone_world[1] + 3 * bearing_1_drone[1]],
    [drone_world[2], drone_world[2] + 3 * bearing_1_drone[2]],
    color="r",
)
ax1.plot(
    [drone_world[0], drone_world[0] + 3 * bearing_2_drone[0]],
    [drone_world[1], drone_world[1] + 3 * bearing_2_drone[1]],
    [drone_world[2], drone_world[2] + 3 * bearing_2_drone[2]],
    color="r",
)   

# PLOT REAL WORLD AXIS
ax.quiver(
    [1],
    [1],
    [1],
    [1],
    [0],
    [0],
    color="g",
    label="REAL eje x",
    normalize=True,
    alpha=0.5,
)
ax.quiver(
    [1],
    [1],
    [1],
    [0],
    [1],
    [0],
    color="r",
    label="REAL eje y",
    normalize=True,
    alpha=0.5,
)
ax.quiver(
    [1],
    [1],
    [1],
    [0],
    [0],
    [1],
    color="b",
    label="REAL eje z",
    normalize=True,
    alpha=0.5,
)
ax1.quiver(
    [1],
    [1],
    [1],
    [ejes_1[0, 0]],
    [ejes_1[1, 0]],
    [ejes_1[2, 0]],
    color="g",
    label="REAL eje x",
    normalize=True,
    alpha=0.5,
)
ax1.quiver(
    [1],
    [1],
    [1],
    [ejes_1[0, 1]],
    [ejes_1[1, 1]],
    [ejes_1[2, 1]],
    color="r",
    label="REAL eje y",
    normalize=True,
    alpha=0.5,
)
ax1.quiver(
    [1],
    [1],
    [1],
    [ejes_1[0, 2]],
    [ejes_1[1, 2]],
    [ejes_1[2, 2]],
    color="b",
    label="REAL eje z",
    normalize=True,
    alpha=0.5,
)

# limits
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()


ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_zlim(-2, 2)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.legend()


ax.view_init(35, -100)
ax1.view_init(35, -100)
plt.show()
