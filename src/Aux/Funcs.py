from cv2 import aruco
import yaml
import numpy as np


class desiredData:
    def __init__(self) -> None:
        self.feature = []
        self.inSphere = []

    def __repr__(self) -> str:
        return f"features: {type(self.feature)} - inSphere: {type(self.inSphere)}"


class actualData:
    def __init__(self) -> None:
        self.feature = []
        self.inSphere = []

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
        temp["bearing"] = np.array(temp["bearing"])

        temp["inv_camera_intrinsic_parameters"] = np.linalg.inv(
            temp["camera_intrinsic_parameters"])
        return temp


def get_aruco(img: np.ndarray) -> tuple:
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    return corners, ids, rejectedImgPoints


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