import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pathlib
import time

font = {'size'   : 12}
matplotlib.rc('font', **font)


# import cv2
# import os

plt.rcParams["figure.autolayout"] = True

np.set_printoptions(precision=5, suppress=True)

MAIN_PATH = pathlib.Path(__file__).parent.absolute()
UP = "\x1B[3A"
CLR = "\x1B[0K"


class Data:
    def __init__(self) -> None:
        """
        @ Attributes
            feature: np.array -> List of the desired features
            inSphere: np.array -> List of the desired features in the sphere
            inNormalPlane: np.array -> List of the desired features in the normal plane
            bearings: np.array -> List of the desired bearings
        """

        self.feature = []
        self.inSphere = []
        self.inNormalPlane = []
        self.bearings = []

        self.distances = []

        self.draw = []


class dictDist:
    def __init__(self, i: int, j: int, dist: float, dist2: float) -> None:
        """
        Class to store the distance between points for GUO control law

        @ Attributes
            i: int -> Index of the first point
            j: int -> Index of the second point
            dist: float -> Distance between the points in the sphere in desired image
            dist2: float -> Distance between the points in the sphere in actual image

        """
        self.i = i
        self.j = j
        self.dist = dist
        self.dist2 = dist2

    def __repr__(self) -> str:
        return f"{self.dist - self.dist2:5f}"


class Plotting3D(plt.Figure):
    def __init__(self, title: str = "3D Plot"):
        super().__init__()
        self.fig = plt.figure(figsize=(5, 5))
        self.fig.set_size_inches(4, 4, forward=False)
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.autoscale(enable=True, axis="both", tight=True)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_aspect("equal")

        # self.ax.set_title(title)

        # self.ax.view_init(elev=5, azim=60)
        # self.ax.view_init(elev=15, azim=45)
        self.ax.view_init(elev=5, azim=75)

    def __call__(
        self,
        points: np.ndarray,
        color: str = "r",
        marker: str = "o",
        label=None,
        z_order=1,
        alpha: float = 1,
    ) -> None:
        """
        Plot the points in 3D

        @ Parameters
            points: np.ndarray -> Points to plot
            color: str -> Color of the points
            marker: str -> Marker of the points
            label: str -> Label of the points
            z_order: int -> Order of the points
            alpha: float -> Transparency of the points
        """
        self.ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=color,
            marker=marker,
            label=f"{label}",
            zorder=z_order,
            alpha=alpha,
        )

    def trajectory(
        self, points: np.ndarray, color: str = "b", label=None, alpha: float = 1
    ) -> None:
        """
        Plot the trajectory of the camera

        @ Parameters
            points: np.ndarray -> Points to plot
            color: str -> Color of the points
            label: str -> Label of the points
            alpha: float -> Transparency of the points
        """
        self.ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            label=label,
            zorder=1,
            alpha=alpha,
        )
        self.ax.plot(
            points[-1, 0],
            points[-1, 1],
            points[-1, 2],
            color=color,
            marker="o",
            alpha=alpha,
        )
        self.ax.plot(
            points[0, 0],
            points[0, 1],
            points[0, 2],
            color=color,
            marker="o",
            alpha=alpha,
        )

    def trajectoryPoints(
        self, points: np.ndarray, color: str = "b", label=None, alpha: float = 1
    ) -> None:
        """
        Plot the trajectory of points

        @ Parameters
            points: np.ndarray (t, n, 3) -> Points to plot
            color: str -> Color of the points
            label: str -> Label of the points
            alpha: float -> Transparency of the points
        """

        # plot trajectory
        for i in range(points.shape[1]):
            self.ax.plot(
                points[:, i, 0],
                points[:, i, 1],
                points[:, i, 2],
                color=color,
                label=label,
                zorder=1,
                alpha=alpha,
            )

    def show(self) -> None:
        """
        Show the plot
        """
        print("\t[INFO] Showing the plot")
        plt.show()


class PlottingImage(plt.Figure):
    def __init__(self, xlim, ylim):
        super().__init__()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.ax.autoscale(enable=True, axis="both", tight=True)

        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.set_aspect("equal")

        # self.ax.set_title("Camera Image")

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        # get off axis
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

    def __call__(
        self,
        points: np.ndarray,
        color: str = "r",
        marker: str = ".",
        label=None,
        z_order=1,
    ) -> None:
        """
        Plot the points in 3D

        @ Parameters
            points: np.ndarray -> Points to plot
            color: str -> Color of the points
            marker: str -> Marker of the points
            label: str -> Label of the points
            z_order: int -> Order of the points
        """
        self.ax.scatter(
            points[:, 0],
            points[:, 1],
            color=color,
            marker=marker,
            label=f"{label}",
            zorder=z_order,
        )

    def trajectory(self, points: np.ndarray, color: str = "b", label=None) -> None:
        """
        Plot the trajectory of the camera

        @ Parameters
            points: np.ndarray -> Points to plot
            color: str -> Color of the points
            label: str -> Label of the points
        """
        for i in range(points.shape[1]):
            self.ax.plot(
                points[:, i, 0], points[:, i, 1], color=color, label=label, zorder=1
            )

    def show(self) -> None:
        """
        Show the plot
        """
        print("\t[INFO] Showing the plot")
        self.ax.legend()
        self.fig.tight_layout()
        self.fig.show()


def sendToSphere(points: np.ndarray, invK: np.ndarray) -> np.ndarray:
    """
    Send the points to the sphere by the Unified Sphere Model Camera Projection Model

    @ Parameters
        points: np.ndarray -> Points in the image
        invK: np.ndarray -> Inverse of the camera intrinsic parameters

    @ Returns
        tuple (np.ndarray, np.ndarray) -> (points in the sphere, points in the normal plane)
    """
    inSphere = []
    inNormalPlane = []
    temp_points = np.concatenate(
        (points, np.ones((points.shape[0], 1))), axis=1, dtype=np.float32
    )
    for i in range(temp_points.shape[0]):
        temp = invK @ temp_points[i]
        inNormalPlane.append(temp[:2])
        inSphere.append(normalize(temp))
    return np.array(inSphere), np.array(inNormalPlane)


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize a vector

    @ Parameters
        x: np.ndarray -> Vector

    @ Returns
        np.ndarray -> Normalized vector
    """
    return x if (norm := np.linalg.norm(x)) == 0 else x / norm


def ortoProj(x: np.ndarray) -> np.ndarray:
    """
    Calculate the ortogonal projection of a vector in R^3

    @ Parameters
        x: np.ndarray -> Vector in R^3

    @ Returns
        np.ndarray -> Orto projection of the vector size 3x3
    """
    temp = x.reshape(3, 1)
    return (np.eye(3) - (temp @ temp.T)) / np.linalg.norm(temp) ** 2


def decomposeSkewSymmetricMatrix(A: np.ndarray) -> np.ndarray:
    """
    Decompose a skew symmetric matrix in a vector

    @ Parameters
        A: np.ndarray -> Skew symmetric matrix

    @ Returns
        np.ndarray -> Vector
    """
    if np.linalg.norm(A + A.T) > 1e-3:
        raise ValueError(
            f"The matrix is not skew symmetric >> \nA: {A}\nA.T: {A.T}\n-A.T + A: {A + A.T}"
        )
    return np.array([A[2, 1], A[0, 2], A[1, 0]]).reshape(-1, 1)


def skewSymmetricMatrix(x: np.ndarray) -> np.ndarray:
    """
    Calculate the skew symmetric matrix of a vector

    @ Parameters
        x: np.ndarray -> Vector

    @ Returns
        np.ndarray -> Skew symmetric matrix
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def e2R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Calculate the rotation matrix from euler angles

    @ Parameters
        roll: float -> Roll angle
        pitch: float -> Pitch angle
        yaw: float -> Yaw angle

    @ Returns
        np.ndarray -> Rotation matrix
    """
    return (
        np.array(
            [
                [
                    np.cos(yaw) * np.cos(pitch),
                    np.cos(yaw) * np.sin(pitch) * np.sin(roll)
                    - np.sin(yaw) * np.cos(roll),
                    np.cos(yaw) * np.sin(pitch) * np.cos(roll)
                    + np.sin(yaw) * np.sin(roll),
                ],
                [
                    np.sin(yaw) * np.cos(pitch),
                    np.sin(yaw) * np.sin(pitch) * np.sin(roll)
                    + np.cos(yaw) * np.cos(roll),
                    np.sin(yaw) * np.sin(pitch) * np.cos(roll)
                    - np.cos(yaw) * np.sin(roll),
                ],
                [
                    -np.sin(pitch),
                    np.cos(pitch) * np.sin(roll),
                    np.cos(pitch) * np.cos(roll),
                ],
            ]
        )
        .astype(np.float32)
        .reshape(3, 3)
    )


def r2E(R: np.ndarray) -> np.ndarray:
    """
    Calculate the euler angles from a rotation matrix

    @ Parameters
        R: np.ndarray -> Rotation matrix

    @ Returns
        np.ndarray -> Euler angles
    """
    if R[2, 0] < 0.995:
        if R[2, 0] > -0.995:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arcsin(-R[2, 0])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = 0
            pitch = np.pi / 2.0
            yaw = -np.arctan2(-R[1, 2], R[1, 1])
    else:
        roll = 0
        pitch = -np.pi / 2.0
        yaw = np.arctan2(-R[1, 2], R[1, 1])

    return np.array([roll, pitch, yaw])

    # if R[2, 0] < 0.995:
    #     # if R[2,0] > -1.0:
    #     if R[2, 0] > -0.995:
    #         roll = np.arctan2(R[2, 1], R[2, 2])
    #         pitch = np.arcsin(-R[2, 0])
    #         yaw = np.arctan2(R[1, 0], R[0, 0])
    #     else:
    #         roll = 0.0
    #         pitch = np.pi / 2.0
    #         yaw = -np.arctan2(-R[1, 2], R[1, 1])

    # else:
    #     # print("WARN: rotation not uniqe")
    #     roll = 0.0
    #     pitch = -np.pi / 2.0
    #     yaw = np.arctan2(-R[1, 2], R[1, 1])

    # return np.array([roll, pitch, yaw])


def decorator_timer(function):
    """
    Decorator to calculate the time of a function

    @ Parameters
        function: function -> Function to calculate the time

    @ Returns
        wrapper: function -> Wrapper function

    @ Usage
        @decorator_timer
        def function():
            pass

        function()
    """

    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = function(*args, **kwargs)
        end = time.time() - t1
        print(f"\n\t[INFO] Function '{function.__name__}' took {end:5f} seconds")
        return result

    return wrapper


def angleVectors(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calculate the angle between two vectors

    @ Parameters
        u: np.ndarray -> Vector
        v: np.ndarray -> Vector

    @ Returns
        float -> Angle between the vectors
    """
    return np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


class WriteFile:
    def __init__(self, path: str) -> None:
        """
        Class to write a file

        @ Attributes
            path: str -> Path to save the file
        """
        self.path = path
        self.file = open(self.path, "w")

    def __call__(self, data: str, opc: int = 0) -> None:
        """
        Write the data in the file

        @ Parameters
            data: str -> Data to write in the file
        """
        self.file.write(data)
        if opc:
            self.file.write("\n")

    def close(self) -> None:
        """
        Close the file
        """
        self.file.close()


def distancesInSphere(
    actualPoints: np.ndarray, desiredPoints: np.ndarray, control: int = 1
) -> np.ndarray:
    """
    Calculate the distances between the points in the sphere

    @ Parameters
        actualPoints: np.ndarray -> Points in the actual image
        desiredPoints: np.ndarray -> Points in the desired image
        control: int -> Control law to use

    @ Returns
        np.ndarray -> Distances between the points in the sphere
    """
    distances = []
    for i in range(actualPoints.shape[0]):
        for j in range(i):
            dp = np.sqrt(2 - 2 * np.dot(actualPoints[i], actualPoints[j]))
            dd = np.sqrt(2 - 2 * np.dot(desiredPoints[i], desiredPoints[j]))
            if dp <= 1e-9 or dd <= 1e-9:
                continue

            distances.append(
                dictDist(
                    i, j, 1 / dp if control == 1 else dp, 1 / dd if control == 1 else dd
                )
            )
    return distances


def guoL(distances: list, actualPoints: np.ndarray, control: int = 1) -> np.ndarray:
    """
    Calculate the L matrix for the GUO control law

    @ Returns
        np.ndarray -> L matrix
    """
    # print(
    #     f"\t[INFO] Calculating L matrix with GUO control law >> {'1/dist' if control == 1 else 'dist'}"
    # )
    n = len(distances)
    L = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        s = -distances[i].dist ** 3 if control == 1 else 1 / distances[i].dist
        pi = actualPoints[distances[i].i].reshape(3, 1)
        pj = actualPoints[distances[i].j].reshape(3, 1)
        L[i, :] = s * (pi.T @ ortoProj(pj) + pj.T @ ortoProj(pi))
    return L


def middlePoint(points: np.ndarray) -> np.ndarray:
    """
    Calculate the middle point of the points

    @ Parameters
        points: np.ndarray -> Points

    @ Returns
        np.ndarray -> Middle point
    """
    middle = []
    for i in range(0, points.shape[0], 4):
        middle.append(normalize(np.mean(points[i : i + 4], axis=0)))
        # middle.append(points[i])
    return -np.array(middle)


def homography(desiredPoints: np.ndarray, actualPoints: np.ndarray) -> np.ndarray:
    """
    Calculate the homography matrix

    @ Parameters
        desiredPoints: np.ndarray -> Points in the desired image
        actualPoints: np.ndarray -> Points in the actual image

    @ Returns
        np.ndarray -> Homography matrix
    """
    # print(f"desiredPoints {desiredPoints}")
    # print(f"actualPoints {actualPoints}")

    mean = np.mean(desiredPoints, axis=0)
    maxstd = max(np.std(desiredPoints, axis=0)) + 1e-9
    # print(f"mean {mean}")
    # print(f"maxstd {maxstd}")

    C1 = np.diag([1 / maxstd, 1 / maxstd, 1])
    C1[0, 2] = -mean[0] / maxstd
    C1[1, 2] = -mean[1] / maxstd
    # print(f"C1 {C1}")
    # print(f"desiredPoints {desiredPoints}")
    desiredPoints = np.hstack([desiredPoints, np.ones((desiredPoints.shape[0], 1))])
    desiredPoints = (C1 @ desiredPoints.T).T
    # print(f"desiredPoints {desiredPoints}")

    mean = np.mean(actualPoints, axis=0)
    maxstd = max(np.std(actualPoints, axis=0)) + 1e-9
    # print(f"mean {mean}")
    # print(f"maxstd {maxstd}")

    C2 = np.diag([1 / maxstd, 1 / maxstd, 1])
    C2[0, 2] = -mean[0] / maxstd
    C2[1, 2] = -mean[1] / maxstd
    # print(f"C2 {C2}")
    # print(f"actualPoints {actualPoints}")
    actualPoints = np.hstack([actualPoints, np.ones((actualPoints.shape[0], 1))])
    actualPoints = (C2 @ actualPoints.T).T
    # print(f"actualPoints {actualPoints}")

    nbr_correspondences = actualPoints.shape[0]
    A = np.zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [
            desiredPoints[i, 0],
            desiredPoints[i, 1],
            1,
            0,
            0,
            0,
            -actualPoints[i, 0] * desiredPoints[i, 0],
            -actualPoints[i, 0] * desiredPoints[i, 1],
            -actualPoints[i, 0],
        ]
        A[2 * i + 1] = [
            0,
            0,
            0,
            desiredPoints[i, 0],
            desiredPoints[i, 1],
            1,
            -actualPoints[i, 1] * desiredPoints[i, 0],
            -actualPoints[i, 1] * desiredPoints[i, 1],
            -actualPoints[i, 1],
        ]

    U, S, V = np.linalg.svd(A)
    H = V[8].reshape((3, 3))

    H = np.linalg.inv(C2) @ H @ C1

    return H / H[2, 2]


def H2Rt(H: np.ndarray) -> tuple:
    # print(f"H: {H}")
    U, S, V = np.linalg.svd(H, full_matrices=True)
    # print(f"U: {U}, S: {S}, V: {V}")

    s1 = S[0] / S[1]
    s3 = S[2] / S[1]

    zeta = s1 - s3
    # print(f"s1: {s1}, s3: {s3}, zeta: {zeta}")

    a1 = np.sqrt(1 - s3**2)
    b1 = np.sqrt(s1**2 - 1)

    a, b = normalize([a1, b1])
    c, d = normalize([1 + s1 * s3, a1 * b1])
    e, f = normalize([-b / s1, -a / s3])

    # print(f"a1: {a1}, b1: {b1}")
    # print(f"a: {a}, b: {b}")
    # print(f"c: {c}, d: {d}")
    # print(f"e: {e}, f: {f}")

    V1 = V[0]
    V3 = V[2]
    # print(f"V1: {V1}")
    # print(f"V3: {V3}")

    n1 = b * V1 - a * V3
    n2 = b * V1 + a * V3

    R1 = U @ np.array([[c, 0, d], [0, 1, 0], [-d, 0, c]]) @ V
    R2 = U @ np.array([[c, 0, -d], [0, 1, 0], [d, 0, c]]) @ V

    t1 = e * V1 + f * V3
    t2 = -e * V1 + f * V3

    if n1[2] < 0:
        n1 = -n1
        t1 = -t1
    if n2[2] < 0:
        n2 = -n2
        t2 = -t2

    if n1[2] > n2[2]:
        R = R1.T
        t = zeta * t1
        n = n1
    else:
        R = R2.T
        t = zeta * t2
        n = n2
    # print("R1", R1.T)
    # print("R2", R2.T)
    return R


def bearingControl(
    actual: Data,
    desired: Data,
    K: np.ndarray,
    Kinv: np.ndarray,
    control: int = 1,
    toplot: list = None,
) -> np.ndarray:
    """
    Calculate the whole control input for the bearing control law

    @ Parameters
        actual: Data -> Actual data
        desired: Data -> Desired data
        K: np.ndarray -> Camera intrinsic parameters
        Kinv: np.ndarray -> Inverse of the camera intrinsic parameters
        control: int -> Control law to use

    @ Returns
        np.ndarray -> L matrix
    """
    U = np.zeros((1, 3), dtype=np.float32)
    v = np.zeros((3, 3), dtype=np.float32)

    for index in range(actual.bearings.shape[0]):
        # H = cv2.findHomography(
        #     actual.feature[index : index + 4], desired.feature[index : index + 4]
        # )[0]
        H = homography(
            actual.feature[index : index + 4], desired.feature[index : index + 4]
        )

        He = Kinv @ H @ K

        R = H2Rt(He)

        # num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
        # tries = [(Rs[i], Ts[i], Ns[i]) for i in range(num) if Ns[i][2] > 0]

        # if len(tries) == 0:
        #     return np.zeros((6, 1))
        # elif len(tries) == 1:
        #     R, T, N = tries[0]
        # else:
        #     R, T, N = tries[0]
        #     for i in range(1, len(tries)):
        #         if Ns[i][2] > N[2]:
        #             R, T, N = tries[i]
        # print(">>>>>>", np.rad2deg(r2E(R)))
        # angles = np.rad2deg(r2E(R))
        # toplot[index][0].append(angles[0])
        # toplot[index][1].append(angles[1])
        # toplot[index][2].append(angles[2])

        U += (
            -ortoProj(actual.bearings[index]) @ (np.eye(3) + R) @ desired.bearings[index] 
            if control == 1
            else actual.bearings[index] - (np.eye(3) + R) / 2 @ desired.bearings[index]
        )
        # U += (
        #     -ortoProj(actual.bearings[index]) @ (np.eye(3) + R) @ desired.bearings[index]
        #     + actual.bearings[index] - (np.eye(3) + R) / 2 @ desired.bearings[index]
        # )
        v += R.T - R
        # print(
        #     f"\n>>first: {actual.bearings[index]} ||: {np.linalg.norm(actual.bearings[index])}",
        #     f"second: {desired.bearings[index]} ||: {np.linalg.norm(desired.bearings[index])}",
        #     f"U -> {U}",
        #     sep="\n",
        # )
    return np.hstack((U, decomposeSkewSymmetricMatrix(v).T)).T
    # return np.zeros((6, 1))


def summaryPlot(
    time: np.ndarray,
    inputControl: np.ndarray,
    pose: np.ndarray,
    error: np.ndarray,
    errorPix: np.ndarray,
    integral: np.ndarray = None,
    CONTROL: tuple = (0, 0),
    PATH = "./",
) -> None:
    # if integral is None:
    #     fig, ax = plt.subplots(3, 1, sharex=True)
    # else:
    #     fig, ax = plt.subplots(4, 1, sharex=True)
    fig = plt.figure(figsize=(15, 5))
    fig.set_size_inches(12, 4, forward=False)
    ax = np.array([fig.subplots(1, 3, sharex=True)])
    ax = ax.flatten()
    [i.autoscale(enable=True, axis="both", tight=False) for i in ax]

    # time vs pose
    # ax[1].plot(time, pose, label=("$x$", "$y$", "$z$", "$roll$", "$pitch$", "$yaw$"))
    # ax[1].legend(loc="center left", ncol=2, bbox_to_anchor=(1, 0.5), shadow=True)
    # ax[1].set_ylabel("Pose")
    # ax[1].set_title("Pose")

    # time vs inputControl
    ax[0].plot(
        time, inputControl, label=("$V_x$", "$V_y$", "$V_z$", "$W_x$", "$W_y$", "$W_z$")
    )
    # ax[0].legend(loc="center left", ncol=2, bbox_to_anchor=(1, 0.5), shadow=True)
    ax[0].legend(loc="best", shadow=True)
    ax[0].set_ylabel("Input control")
    # ax[0].set_title("Input control")

    # time vs error
    ax[1].plot(
        time,
        error[:, :6],
        label=(
            "Error x",
            "Error y",
            "Error z",
            "Error pitch",
            "Error yaw",
            "Error roll",
        )
        if len(error.shape) > 1
        else ("Error (c)"),
    )
    ax[1].plot(time, errorPix, label="Error (px)")
    # ax[1].legend(loc="center left", ncol=2, bbox_to_anchor=(1, 0.5), shadow=True)
    ax[1].legend(loc="best", shadow=True)
    ax[1].set_ylabel("Errors")
    # ax[1].set_title("Pose Error")

    ax[2].plot(
        time,
        error[:, 6:],
        label=(
            "Error x",
            "Error y",
            "Error z",
        )
        if CONTROL[0] == 1
        else tuple([f"$s_{i+1}$" for i in range(len(error[0, 6:]))])
        if len(error.shape) > 1
        else ("Error (c)"),
    )
    ax[2].legend(loc="best", shadow=True)
    ax[2].set_ylabel("Errors")
    # if CONTROL[0] is 1 Bearing Error but, if control[1] is 1 + Orthogonal Projection or else + Difference
    # else if CONTROL[0] is 0 Distance Error but, if control[1] is 1 + Orthogonal Projection or else + Difference
    # ax[2].set_title(
    #     (
    #         "Bearing Error "
    #         + ("Orthogonal Projection" if CONTROL[1] == 1 else "Difference")
    #         if CONTROL[0]
    #         else ("Distance Error " + ("1/dist" if CONTROL[1] == 1 else "dist"))
    #     )
    # )

    # # time vs errorPix
    # if integral is not None:
    #     # time vs integral
    #     ax[3].plot(time, integral, label=("Vx", "Vy", "Vz", "Wx", "Wy", "Wz"))
    #     ax[3].legend(loc="center left", ncol=2, bbox_to_anchor=(1, 0.5), shadow=True)
    #     ax[3].set_ylabel("Integral")
        # ax[3].set_title("Integral")

    [i.set_xlabel("Time [s]") for i in ax]
    fig.tight_layout()
    fig.savefig(
        f"{PATH}/out/PlottingError{'Leader' if not CONTROL[0] else 'Follower'}{('OrthogonalProj' if CONTROL[1] == 1 else 'Difference') if CONTROL[0] else ('1_dist' if CONTROL[1] == 1 else 'dist')}.svg",
        format="svg",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def pointsMove(t: float, fx, fy, fz, n: int = 8) -> tuple:
    """
    Calculate the points to move

    @ Parameters
        t: float -> X position
        fx -> Function to calculate the x position
        fy -> Function to calculate the y position
        fz -> Function to calculate the z position
        n: int -> Number of points to move

    @ Returns
        tuple -> Points to move
    """
    N = np.zeros((n, 3), dtype=np.float32)
    N[:, 0] = fx(t)
    N[:, 1] = fy(t)
    N[:, 2] = fz(t)
    return N
