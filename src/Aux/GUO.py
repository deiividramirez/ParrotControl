from pathlib import Path
import numpy as np
import time
import cv2


PATH = Path(__file__).parent.absolute().parent.absolute().parent.absolute()


if __name__ == "__main__":
    # load python file from src/Aux/Funcs.py
    from Funcs import *
else:
    # load python file from Aux/Funcs.py
    from src.Aux.Funcs import *


class GUO:
    def __init__(
        self, img_desired: np.ndarray, drone_id: int, RT: np.ndarray = np.eye(3, 3)
    ) -> None:
        """
        __init__ function for the GUO class

        This class makes possible to use the control law proposed in the paper
        "Image-based estimation, planning, and control for high-speed flying
        through multiple openings" by Guo et al. (2019).

        This is an Image Based Visual Servoing (IBVS) method, which means that
        the control law is based on the image of the drone, not in the state of
        the drone. Here, the control uses some invariant features of the image
        to make the drone move to the desired position by decoupling the
        translational and rotational components of the control law.

        @Params:
            img_desired: np.ndarray -> A (n,3) matrix with the desired image
            drone_id: int -> A flag to know which drone is going to be used
            RT: np.ndarray -> A (3,3) matrix with the rotation and translation
                for changing the reference frame from the camera to the drone

        @Returns:
          None

        """
        self.img_desired = img_desired
        self.img_desired_gray = cv2.cvtColor(img_desired, cv2.COLOR_BGR2GRAY)
        self.drone_id = drone_id
        self.rotAndTrans = RT
        self.yaml = load_yaml(PATH, drone_id)

        n = len(self.yaml["seguimiento"])
        quant = (n - 1) * n // 2
        if n not in (1, 4):
            raw = input(
                f"[INFO] Using {n} ArUco markers for control law. That is {quant} distances. Continue? (y/n): "
            )
            if raw.lower() != "y":
                exit()
        elif n == 4:
            print("[INFO] Using the 4 most distant points of the 4 ArUco markers")

        print(
            f"[INFO] Control law {'1/dist' if self.yaml['control'] == 1 else 'dist'} with {quant} distances\n"
        )

        if self.getDesiredData() < 0:
            print("Desired ArUco not found")
            exit()

        self.storeImage = None
        self.initTime = time.time()
        self.actualTime = time.time()
        self.error = np.zeros((1, 6))
        self.errorNorm = 0
        self.errorPix = 0

        self.gain_v = adaptativeGain(
            self.yaml["gain_v_kp_ini"],
            self.yaml["gain_v_kp_max"],
            self.yaml["l_prime_v_kp"],
        )

        self.gain_w = adaptativeGain(
            self.yaml["gain_w_kp_ini"],
            self.yaml["gain_w_kp_max"],
            self.yaml["l_prime_w_kp"],
        )

        self.file_vel_x = open(PATH / "out" / f"drone_{drone_id}_vel_x.txt", "w+")
        self.file_vel_y = open(PATH / "out" / f"drone_{drone_id}_vel_y.txt", "w+")
        self.file_vel_z = open(PATH / "out" / f"drone_{drone_id}_vel_z.txt", "w+")
        self.file_vel_yaw = open(PATH / "out" / f"drone_{drone_id}_vel_yaw.txt", "w+")
        self.file_error = open(PATH / "out" / f"drone_{drone_id}_error.txt", "w+")
        self.file_errorPix = open(PATH / "out" / f"drone_{drone_id}_errorPix.txt", "w+")
        self.file_time = open(PATH / "out" / f"drone_{drone_id}_time.txt", "w+")
        self.file_v_kp = open(PATH / "out" / f"drone_{drone_id}_v_kp.txt", "w+")
        self.file_v_ki = open(PATH / "out" / f"drone_{drone_id}_v_ki.txt", "w+")
        self.file_w_kp = open(PATH / "out" / f"drone_{drone_id}_w_kp.txt", "w+")
        # self.file_w_ki = open(PATH / "out" / f"drone_{drone_id}_w_ki.txt", "w+")
        self.file_int = open(PATH / "out" / f"drone_{drone_id}_int.txt", "w+")

        self.file_v_kp.write("0.0\n")
        self.file_v_ki.write("0.0\n")
        self.file_w_kp.write("0.0\n")
        self.file_time.write("0.0\n")
        self.file_vel_x.write("0.0\n")
        self.file_vel_y.write("0.0\n")
        self.file_vel_z.write("0.0\n")
        self.file_vel_yaw.write("0.0\n")
        self.file_error.write("0.0\n")
        self.file_errorPix.write("0.0\n")

    def __name__(self) -> str:
        return "GUO: 1/dist" if self.yaml["control"] == 1 else "GUO: dist"

    def getDesiredData(self) -> int:
        """
        This function get the desired data from the desired image, send the points to
        an sphere with the unified model of camera

        @Params:
          None

        @Returns:
          int -> A flag to know if the aruco was found or not
        """
        self.desiredData = desiredData()
        temp = get_aruco(self.img_desired_gray, 4)

        for seg in self.yaml["seguimiento"]:
            if temp[1] is not None and seg in temp[1]:
                index = np.argwhere(temp[1] == seg)[0][0]
                self.desiredData.feature.append(temp[0][index][0])
            else:
                print("ArUco not found")
                return -1
        self.desiredData.feature = np.array(
            self.desiredData.feature, dtype=np.int32
        ).reshape(-1, 2)

        if len(self.yaml["seguimiento"]) == 4:
            self.desiredData.feature = np.array(
                [self.desiredData.feature[i].copy() for i in [0, 5, 11, 14]]
            )

        self.desiredData.inSphere, self.desiredData.inNormalPlane = sendToSphere(
            self.desiredData.feature, self.yaml["inv_camera_intrinsic_parameters"]
        )

        return 0

    def getActualData(self, actualImage: np.ndarray, imgAruco: tuple) -> int:
        """
        This function get the actual data from the actual image, send the points to
        an sphere with the unified model of camera

        @Params:
          actualImage: np.ndarray -> A (n,3) matrix with the actual image

        @Returns:
          int -> A flag to know if the aruco was found or not
        """
        self.actualData = actualData()
        # imgAruco = get_aruco(actualImage)

        for seg in self.yaml["seguimiento"]:
            if imgAruco[1] is not None and seg in imgAruco[1]:
                index = np.argwhere(imgAruco[1] == seg)[0][0]
                self.actualData.feature.append(imgAruco[0][index][0])
            else:
                # print("ArUco not found")
                return -1
        self.actualData.feature = np.array(
            self.actualData.feature, dtype=np.int32
        ).reshape(-1, 2)

        if len(self.yaml["seguimiento"]) == 4:
            self.actualData.feature = np.array(
                [self.actualData.feature[i] for i in [0, 5, 11, 14]]
            )

        self.actualData.inSphere, self.actualData.inNormalPlane = sendToSphere(
            self.actualData.feature, self.yaml["inv_camera_intrinsic_parameters"]
        )

        return 0

    @decorator_timer
    def getVels(self, actualImage: np.ndarray, imgAruco: tuple) -> np.ndarray:
        """
        This function returns the velocities of the drones in the drone's frame
        It will use the desired image and the actual image to calculate the velocities
        with the GUO proposed method in the paper "Image-based estimation, planning,
        and control for high-speed flying through multiple openings".

        @Params:
          actualImage: np.ndarray -> A (m,n) matrix with the actual image of the drone's camera

        @Returns:
          vels: np.ndarray -> A (6x1) array for the velocities of the drone in the drone's frame
        """
        # if (
        #     self.storeImage is not None
        #     and actualImage is not None
        #     and np.array_equal(self.storeImage, actualImage)
        # ):
        #     print("Same image")
        #     return self.input
        # else:
        #     self.storeImage = actualImage
        # self.storeImage = actualImage

        if self.getActualData(actualImage, imgAruco) < 0:
            print("[ERROR] Some ArUco's were not found")
            self.input = np.zeros((6,))
            self.save()
            return self.input

        self.distances, self.error = self.getDistances(
            self.actualData.inSphere, self.desiredData.inSphere
        )
        print(" Distances: ", len(self.distances), self.distances)

        self.errorNorm = np.linalg.norm(self.error, ord=1)

        self.L = self.laplacianGUO(self.actualData.inSphere, self.distances)
        self.Lp = np.linalg.pinv(self.L)

        self.vels = np.concatenate(
            (self.Lp @ self.error, self.rotationControl()), axis=0
        )

        self.input = np.concatenate(
            (self.rotAndTrans @ self.vels[:3], self.rotAndTrans @ self.vels[3:]),
            axis=0,
            dtype=np.float32,
        ).reshape((6,))

        self.input[:3] = self.gain_v(self.errorNorm) * self.input[:3]
        self.input[3:] = self.gain_w(self.errorNorm) * self.input[3:]
        # self.input[2] *= 2

        self.input = np.clip(self.input, -self.yaml["max_vel"], self.yaml["max_vel"])

        if self.yaml["vels"] == 1:
            print("Input before %: ", self.input)
            self.input[:3] = self.input[:3] * 100 / self.yaml["max_vel"]
            self.input = np.clip(self.input, -60, 60)
            self.input[2] = np.clip(self.input[2], -40, 40)
            print("Input  after %: ", self.input)

        self.save()
        return self.input

    def save(self):
        # print("[INFO] Saving data")
        self.actualTime = time.time() - self.initTime
        try:
            self.file_vel_x.write(f"{self.input[0]}\n")
            self.file_vel_y.write(f"{self.input[1]}\n")
            self.file_vel_z.write(f"{self.input[2]}\n")
            self.file_vel_yaw.write(f"{self.input[5]}\n")
            self.file_time.write(f"{self.actualTime}\n")
            self.file_error.write(f"{self.errorNorm}\n")
            self.file_errorPix.write(f"{self.errorPix}\n")

            self.file_v_kp.write(f"{self.gain_v.gain}\n")
            self.file_w_kp.write(f"{self.gain_w.gain}\n")

            # print(f"x: {self.input[0]}",
            #         f"y: {self.input[1]}",
            #         f"z: {self.input[2]}",
            #         f"yaw: {self.input[5]}",
            #         f"error: {error:.5f}",
            #         f"time: {self.actualTime:.2f}",
            #         sep="\t")

            print(
                f" Error control: {self.errorNorm:3f}\nError pixels {self.errorPix:3f}\nLambda v: {self.gain_v.gain:3f}\nLambda w: {self.gain_w.gain:3f}\n",
            )

        except Exception as e:
            print("[ERROR] Error writing in file: ", e)

    def rotationControl(self) -> np.ndarray:
        """
        This function returns the angular velocities of the drone in the drone's frame

        @Params:
            None

        @Returns:
            np.ndarray -> A (3x1) array for the angular velocities of the drone in the drone's frame
        """
        L = np.zeros((2 * self.actualData.inNormalPlane.shape[0], 3))
        for i in range(2 * self.actualData.inNormalPlane.shape[0]):
            u, v = self.actualData.inNormalPlane[i // 2]
            if i % 2 == 0:
                L[i, :] = np.array([u * v, -(1 + u**2), v])
            else:
                L[i, :] = np.array([1 + v**2, -u * v, -u])
        try:
            Linv = np.linalg.pinv(L)
            Error = (
                self.actualData.inNormalPlane - self.desiredData.inNormalPlane
            ).reshape(-1, 1)
            self.errorPix = np.linalg.norm(Error)
            return -Linv @ Error
        except Exception as e:
            print("[ERROR] Error in rotationControl: ", e)
            return np.zeros((3, 1))

    def getDistances(
        self, pointsActual: np.ndarray, pointsDesired: np.ndarray
    ) -> tuple:
        """
        This function returns the Euclidean distance between the points in the sphere with the unified model
        of camera with the actual and desired points obtained from the image.

        @Params:
          pointsActual: np.ndarray -> A (n,3) matrix with the actual points in the sphere
          pointsDesired: np.ndarray -> A (n,3) matrix with the desired points in the sphere

        @Returns:
          tuple -> A tuple with the distances and the error

        """
        distances, error = [], []
        for i in range(pointsActual.shape[0]):
            for j in range(i):
                # for j in range(pointsActual.shape[1]):
                if i != j:
                    dist = np.sqrt(2 - 2 * np.dot(pointsDesired[i], pointsDesired[j]))
                    dist2 = np.sqrt(2 - 2 * np.dot(pointsActual[i], pointsActual[j]))
                    if dist <= 1e-9 or dist2 <= 1e-9:
                        continue

                    distances.append(
                        dictDist(i, j, 1 / dist, 1 / dist2)
                        if self.yaml["control"] == 1
                        else dictDist(i, j, dist, dist2)
                    )

        # distances = sorted(distances, key=lambda x: x.dist, reverse=True)
        error = [distance.dist - distance.dist2 for distance in distances]
        return distances, np.array(error).reshape(len(error), 1)

    def laplacianGUO(self, pointsSphere: np.ndarray, distances: dictDist):
        """
        This function returns the laplacian matrix for the GUO method in the paper "Image-based estimation, planning,
        and control for high-speed flying through multiple openings".

        @Params:
          pointsSphere: np.ndarray -> A (n,3) matrix with the points in the sphere
          distances: dictDist -> A list of dictionaries with the distances between the points in the sphere

        @Returns:
          L: np.ndarray -> A (n,3) matrix with the laplacian matrix
        """
        n = len(distances)
        L = np.zeros((n, 3))

        for i in range(n):
            s = (
                -distances[i].dist2 ** 3
                if self.yaml["control"] == 1
                else 1 / distances[i].dist2
            )

            p_i = pointsSphere[distances[i].i].reshape(3, 1)
            p_j = pointsSphere[distances[i].j].reshape(3, 1)

            L[i, :] = s * (p_i.T @ ortoProj(p_j) + p_j.T @ ortoProj(p_i))
        return L

    def close(self):
        self.file_errorPix.close()
        self.file_vel_yaw.close()
        self.file_error.close()
        self.file_vel_x.close()
        self.file_vel_y.close()
        self.file_vel_z.close()
        self.file_time.close()
        self.file_v_kp.close()
        self.file_w_kp.close()


if __name__ == "__main__":
    img = cv2.imread(f"{PATH}/data/desired_1.jpg")
    control = GUO(img, 1)

    print(
        control.getVels(
            cv2.imread(f"{PATH}/data/desired_1.jpg"),
            get_aruco(cv2.imread(f"{PATH}/data/desired_1.jpg"), 4),
        )
    )

    print(
        control.getVels(
            cv2.imread(f"{PATH}/data/desired_2.jpg"),
            get_aruco(cv2.imread(f"{PATH}/data/desired_2.jpg"), 4),
        )
    )

    # Close files
    control.close()
