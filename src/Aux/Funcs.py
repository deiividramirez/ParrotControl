from functools import wraps
from cv2 import aruco
import numpy as np
import time
import yaml
import cv2


class desiredData:
    def __init__(self) -> None:
        self.feature = []
        self.inSphere = []
        self.inNormalPlane = []
        self.bearings = []

    def __repr__(self) -> str:
        return f"features: {type(self.feature)} - inSphere: {type(self.inSphere)}"


class actualData:
    def __init__(self) -> None:
        self.feature = []
        self.inSphere = []
        self.inNormalPlane = []
        self.bearings = []

    def __repr__(self) -> str:
        return f"features: {type(self.feature)} - inSphere: {type(self.inSphere)}"


class dictDist:
    def __init__(self, i: int, j: int, dist: float, dist2: float) -> None:
        self.i = i
        self.j = j
        self.dist = dist
        self.dist2 = dist2

    def __repr__(self) -> str:
        return (
            # f"i: {self.i} <-> j: {self.j}: d1 -> {self.dist:5f} - d2 -> {self.dist2:5f}"
            # f"i: {self.i} <-> j: {self.j}: d2-d1 -> {self.dist2-self.dist:5f}"
            # f"p_{self.i}, p_{self.j} -> {self.dist:5f} - d2 -> {self.dist2:5f} := {self.dist2-self.dist:5f}\n"
            # f"{self.dist:5f}* (=) {self.dist2:5f}"
            f"{self.dist - self.dist2:5f}"
        )


class drawArucoClass:
    def __init__(self) -> None:
        self.img = None

    def drawAruco(self, img: np.ndarray, info: tuple) -> np.ndarray:
        self.img = img.copy()
        drawArucoPoints(self.img, info)

    def drawNew(self, info: np.ndarray, color: tuple = (0, 255, 0)):
        for i in range(info.shape[0]):
            cv2.circle(self.img, tuple(info[i]), 3, color, -1)


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

    def __call__(self, error: float, axis=None) -> float:
        try:
            if axis is None:
                self.gain = self.getGain(error)
            elif axis == 0:
                self.gain_0 = self.getGain(error)
                self.gain = self.last_gain_0 = self.gain_0
            elif axis == 1:
                self.gain_1 = self.getGain(error)
                self.gain = self.last_gain_1 = self.gain_1
            elif axis == 2:
                self.gain_2 = self.getGain(error)
                self.gain = self.last_gain_2 = self.gain_2
        except:
            if axis is None:
                self.gain = self.last_gain
            elif axis == 0:
                self.gain_0 = self.last_gain_0
            elif axis == 1:
                self.gain_1 = self.last_gain_1
            elif axis == 2:
                self.gain_2 = self.last_gain_2

        self.last_gain = self.gain
        return self.gain

    def getGain(self, error: float) -> float:
        if self.gain_init != self.gain_max:
            return self.gain_init + (self.gain_max - self.gain_init) * np.exp(
                -error * self.l_prime / (self.gain_max - self.gain_init)
            )
        else:
            return self.gain_init


def load_yaml(PATH, drone_id) -> dict:
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
    with open(f"{PATH}/config/general.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_aruco(img: np.ndarray, n: int = 6) -> tuple:
    if n == 6:
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    if n == 4:
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

    parameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    return np.int32(corners), ids


# def drawArucoSimple(img: np.ndarray, info: tuple) -> np.ndarray:
#     temp_img = img.copy()
#     corners, ids, _ = info
#     return aruco.drawDetectedMarkers(temp_img, corners, ids)


def drawArucoPoints(
    img: np.ndarray, info: tuple, color: tuple = (0, 0, 255)
) -> np.ndarray:
    corners, ids = info
    for i in range(len(ids)):
        for j in range(4):
            cv2.circle(img, tuple(corners[i][0][j]), 3, color, -1)
    return img


def sendToSphere(points: np.ndarray, invK: np.ndarray) -> np.ndarray:
    inSphere = []
    inNormalPlane = []
    temp_points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    for i in range(temp_points.shape[0]):
        temp = invK @ temp_points[i]
        inNormalPlane.append(temp[:2])
        inSphere.append(normalize(temp))
        # print(i, temp_points[i], temp)
        # print(inNormalPlane[-1])
        # print(inSphere[-1])
    return np.array(inSphere), np.array(inNormalPlane)


def normalize(x: np.ndarray) -> np.ndarray:
    return x if (norm := np.linalg.norm(x)) == 0 else x / norm


def ortoProj(x: np.ndarray) -> np.ndarray:
    temp = x.reshape(3, 1)
    return (np.eye(3) - (temp @ temp.T)) / np.linalg.norm(temp) ** 2


def decorator_timer(function):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = function(*args, **kwargs)
        end = time.time() - t1
        print(f"\n\t[INFO] Function '{function.__name__}' took {end:5f} seconds")
        return result

    return wrapper
