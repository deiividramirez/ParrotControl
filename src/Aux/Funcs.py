from cv2 import aruco
import numpy as np
import yaml
import cv2


class desiredData:
    def __init__(self) -> None:
        self.feature = []
        self.inSphere = []
        self.bearings = []

    def __repr__(self) -> str:
        return f"features: {type(self.feature)} - inSphere: {type(self.inSphere)}"


class actualData:
    def __init__(self) -> None:
        self.feature = []
        self.inSphere = []
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
        return f"i: {self.i} <-> j: {self.j}: d1 -> {self.dist:5f} - d2 -> {self.dist2:5f}"


def load_yaml(PATH, drone_id) -> dict:
    with open(f"{PATH}/config/drone_{drone_id}.yaml", "r") as f:
        temp = yaml.load(f, Loader=yaml.FullLoader)
        temp["camera_intrinsic_parameters"] = np.array(
            temp["camera_intrinsic_parameters"]).reshape(3, 3)
        temp["seguimiento"] = np.array(temp["seguimiento"])
        temp["bearings"] = np.array(temp["bearings"], dtype=np.float32).reshape(-1, 3)

        temp["inv_camera_intrinsic_parameters"] = np.linalg.inv(
            temp["camera_intrinsic_parameters"])
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
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    return np.int32(corners), ids, rejectedImgPoints

def drawAruco(img: np.ndarray, info: tuple) -> np.ndarray:
    temp_img = img.copy()
    corners, ids, rejectedImgPoints = info
    return aruco.drawDetectedMarkers(temp_img, corners, ids)
    

def drawArucoPoints(img: np.ndarray, info: tuple) -> np.ndarray:
    temp_img = img.copy()
    # temp_img = img
    corners, ids, rejectedImgPoints = info
    for i in range(len(ids)):
        for j in range(4):
            cv2.circle(temp_img, tuple(corners[i][0][j]), 2, (0, 0, 255), -1)
    return temp_img

def sendToSphere(points: np.ndarray, invK: np.ndarray) -> np.ndarray:
    temp = []
    temp_points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    for i in range(temp_points.shape[0]):
        temp.append(normalize(invK @ temp_points[i]))
    return np.array(temp)


def normalize(x: np.ndarray) -> np.ndarray:
    return x if (norm := np.linalg.norm(x)) == 0 else x / norm

def ortoProj(x: np.ndarray) -> np.ndarray:
    temp = x.reshape(3, 1)
    return np.eye(3) - temp @ temp.T