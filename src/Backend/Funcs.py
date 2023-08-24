from functools import wraps
from cv2 import aruco
import numpy as np
import time
import yaml
import cv2

# Set the numpy print options for all the project
np.set_printoptions(precision=5, suppress=True)


class desiredData:
    def __init__(self) -> None:
        """
        Class to store the desired data

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

        self.draw = []

    def __repr__(self) -> str:
        return f"features: {type(self.feature)} - inSphere: {type(self.inSphere)}"


class actualData:
    def __init__(self) -> None:
        """
        Class to store the actual data

        @ Attributes
            feature: np.array -> List of the actual features
            inSphere: np.array -> List of the actual features in the sphere
            inNormalPlane: np.array -> List of the actual features in the normal plane
            bearings: np.array -> List of the actual bearings
        """
        self.feature = []
        self.inSphere = []
        self.inNormalPlane = []
        self.bearings = []

        self.draw = []

    def __repr__(self) -> str:
        return f"features: {type(self.feature)} - inSphere: {type(self.inSphere)}"


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


class drawArucoClass:
    def __init__(self) -> None:
        """
        Class to draw the aruco points

        @ Attributes
            img: np.ndarray -> Image
        """
        self.img = None
        self.maxPoints = 150
        self.colors = np.random.randint(0, 255, (self.maxPoints, 3)).tolist()
        for i in range(self.maxPoints):
            self.colors[i] = tuple(self.colors[i])

    def drawAruco(self, img: np.ndarray, info: tuple) -> None:
        """
        Draw the aruco points

        @ Parameters
            img: np.ndarray -> Image
            info: tuple -> (corners, ids)

        @ Returns
            None
        """
        self.img = img.copy()
        drawArucoPoints(self.img, info)

    def drawNew(self, info: np.ndarray, color: tuple = None) -> None:
        """
        Draw the new points without replace the image

        @ Parameters
            info: np.ndarray -> New points
            color: tuple (optional) -> Color of the points (BGR)

        @ Returns
            None
        """
        if color is None and info.shape[0] < self.maxPoints:
            for i in range(info.shape[0]):
                cv2.circle(self.img, tuple(info[i]), 5, self.colors[i], -1)
        elif info.shape[0] > self.maxPoints:
            for i in range(info.shape[0]):
                cv2.circle(self.img, tuple(info[i]), 5, (0, 0, 255), -1)
        else:
            for i in range(info.shape[0]):
                cv2.circle(self.img, tuple(info[i]), 5, color, -1)


class adaptativeGain:
    def __init__(self, gain_init: float, gain_max: float, l_prime: float) -> None:
        """
        Adaptative gain function

        Parameters
            gain_init: float -> Minimum gain
            gain_max: float -> Maximum gain
            l_prime: float -> Constant

        Returns when called
            gain: float -> Gain

        Example
            >>> gain = adaptativeGain(1, 0.1, 0.1)
            >>> gain(0.01)
        """
        self.gain_max = gain_max
        self.gain_init = gain_init
        self.l_prime = l_prime

        if (l_prime > 0) and (gain_init > gain_max):
            raise ValueError(
                "The gain_init must be less than gain_max when l_prime > 0 due to the exponential function"
            )

        if (l_prime < 0) and (gain_init < gain_max):
            raise ValueError(
                "The gain_init must be greater than gain_max when l_prime < 0 due to the exponential function"
            )

        self.gain = self.last_gain = self.gain_init
        self.gain_0 = self.last_gain_0 = self.gain
        self.gain_1 = self.last_gain_1 = self.gain
        self.gain_2 = self.last_gain_2 = self.gain

    def __call__(self, error: (float or np.ndarray)) -> float:
        if isinstance(error, np.ndarray):
            self.gain_0 = self.getGain(error[0])
            self.gain_1 = self.getGain(error[1])
            self.gain_2 = self.getGain(error[2])
            self.gain = np.array([[self.gain_0], [self.gain_1], [self.gain_2]])
        else:
            self.gain = self.getGain(error)

        self.last_gain = self.gain
        return self.gain

    def getGain(self, error: float) -> float:
        if self.gain_init != self.gain_max:
            return self.gain_init + (self.gain_max - self.gain_init) * np.exp(
                -error * self.l_prime / (self.gain_max - self.gain_init)
            )
        else:
            return self.gain_init


def load_yaml(PATH: str, drone_id: int = 1) -> dict:
    """
    Load the yaml file for the drone with id drone_id

    @ Parameters
        PATH: str -> Path to the yaml file
        drone_id: int -> Id of the drone

    @ Returns
        dict -> Dictionary with the yaml file
    """
    with open(f"{PATH}/config/drone_{drone_id}.yaml", "r") as f:
        temp = yaml.load(f, Loader=yaml.FullLoader)
        temp["camera_intrinsic_parameters"] = np.array(
            temp["camera_intrinsic_parameters"]
        ).reshape(3, 3)
        temp["seguimiento"] = np.array(temp["seguimiento"])
        # temp["bearings"] = np.array(temp["bearings"], dtype=np.float32).reshape(-1, 3)

        temp["inv_camera_intrinsic_parameters"] = np.linalg.inv(
            temp["camera_intrinsic_parameters"]
        )
        return temp


def loadGeneralYaml(PATH) -> dict:
    """
    Load the general yaml file

    @ Parameters
        PATH: str -> Path to the yaml file

    @ Returns
        dict -> Dictionary with the yaml file
    """
    with open(f"{PATH}/config/general.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_aruco(img: np.ndarray, n: int = 6) -> tuple:
    """
    Get the aruco corners and ids

    @ Parameters
        img: np.ndarray -> Image
        n: int -> Aruco dictionary size

    @ Returns
        tuple -> (corners: np.int32, ids: np.int32)

    """
    if n == 6:
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    if n == 4:
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

    parameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    return np.int32(corners), ids


def drawArucoPoints(
    img: np.ndarray, info: tuple, color: tuple = (0, 0, 255)
) -> np.ndarray:
    """
    Draw the aruco points

    @ Parameters
        img: np.ndarray -> Image
        info: tuple -> (corners, ids)
        color: tuple (optional) -> Color of the points (BGR)

    @ Returns
        img: np.ndarray -> Image with the aruco points
    """
    corners, ids = info
    for i in range(len(ids)):
        for j in range(4):
            cv2.circle(img, tuple(corners[i][0][j]), 5, color, -1)
    return img


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
    temp_points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
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
    # double roll = 0, pitch = 0, yaw = 0;
    #     if (R.at<double>(2, 0) < 1)
    #     {
    #             if (R.at<double>(2, 0) > -1)
    #             {
    #                     roll = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
    #                     pitch = asin(-R.at<double>(2, 0));
    #                     yaw = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    #             }
    #             else
    #             {
    #                     // Not a unique solution:  roll - yaw = atan2(-m12,m11)
    #                     roll = -atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
    #                     pitch = -M_PI / 2.0;
    #                     yaw = 0;
    #             }
    #     }
    #     else
    #     {
    #             // Not a unique solution:  roll + yaw = atan2(-m12,m11)
    #             roll = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
    #             pitch = M_PI / 2.0;
    #             yaw = 0;
    #     }

    if R[2, 0] < 1:
        if R[2, 0] > -1:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arcsin(-R[2, 0])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = -np.arctan2(-R[1, 2], R[1, 1])
            pitch = -np.pi / 2.0
            yaw = 0
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.pi / 2.0
        yaw = 0

    return np.array([roll, pitch, yaw])


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


def findHomography(desiredPoints: np.ndarray, actualPoints: np.ndarray) -> np.ndarray:
    """
    Calculate the homography matrix

    @ Parameters
        desiredPoints: np.ndarray -> Points in the desired image
        actualPoints: np.ndarray -> Points in the actual image

    @ Returns
        np.ndarray -> Homography matrix
    """

    mean = np.mean(desiredPoints, axis=0)
    maxstd = max(np.std(desiredPoints, axis=0)) + 1e-9
    C1 = np.diag([1 / maxstd, 1 / maxstd, 1])
    C1[0, 2] = -mean[0] / maxstd
    C1[1, 2] = -mean[1] / maxstd
    desiredPoints = np.hstack([desiredPoints, np.ones((desiredPoints.shape[0], 1))])
    desiredPoints = (C1 @ desiredPoints.T).T

    mean = np.mean(actualPoints, axis=0)
    maxstd = max(np.std(actualPoints, axis=0)) + 1e-9
    C2 = np.diag([1 / maxstd, 1 / maxstd, 1])
    C2[0, 2] = -mean[0] / maxstd
    C2[1, 2] = -mean[1] / maxstd
    actualPoints = np.hstack([actualPoints, np.ones((actualPoints.shape[0], 1))])
    actualPoints = (C2 @ actualPoints.T).T

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

    try:
        U, S, V = np.linalg.svd(A)
    except Exception as e:
        print(f"[ERROR] Cannot calculate the homography matrix >> {e}")
        return None

    H = V[8].reshape((3, 3))
    H = np.linalg.inv(C2) @ H @ C1

    return H / H[2, 2]


def H2Rt(H: np.ndarray) -> tuple:
    """
    Decompose the homography matrix in R, t, and n  
    
    @ Parameters
        H: np.ndarray -> Homography matrix

    @ Returns
        tuple -> (R: np.ndarray, t: np.ndarray, n: np.ndarray)
    """
    try:
        U, S, V = np.linalg.svd(H, full_matrices=True)

        s1 = S[0] / S[1]
        s3 = S[2] / S[1]

        zeta = s1 - s3

        a1 = np.sqrt(1 - s3**2)
        b1 = np.sqrt(s1**2 - 1)

        a, b = normalize([a1, b1])
        c, d = normalize([1 + s1 * s3, a1 * b1])
        e, f = normalize([-b / s1, -a / s3])


        V1 = V[0]
        V3 = V[2]

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
        return R, t, n
        
    except Exception as e:
        print(f"[ERROR] Cannot decompose the homography matrix >> {e}")
        return (None, None, None)
