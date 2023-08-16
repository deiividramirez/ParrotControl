from pathlib import Path
import numpy as np
import math
import time
import cv2


PATH = Path(__file__).parent.absolute().parent.absolute().parent.absolute()


if __name__ == "__main__":
    # load python file from src/Backend/Funcs.py
    from Funcs import *
else:
    # load python file from Backend/Funcs.py
    from src.Backend.Funcs import *


class BearingOnly:
    def __init__(
        self, img_desired: np.ndarray, drone_id: int, RT: np.ndarray = np.eye(3, 3)
    ) -> None:
        """
        __init__ function for the BearingOnly class

        This class makes possible to use the control law proposed in the paper
        "Translational and scaling formation maneuver control via
        bearing-based approach" by Shiyu Zhao, Daniel Zelazo

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
        if n <= 1:
            print("[ERROR] Only one ArUco is not allowed")
            exit()
        elif n > 2:
            raw = input(
                f"[INFO] Using {n} ArUco markers for {n} bearing measurements. Continue? (y/n): "
            )
            if raw.lower() != "y":
                exit()
        print(
            f"[INFO] Control law {'(-P_gij * gij*)' if self.yaml['control'] == 1 else '(gij - gij*)'}"
        )

        if self.getDesiredData() < 0:
            print("Desired ArUco not found")
            exit()

        self.storeImage = None
        self.initTime = time.time()
        self.actualTime = 0
        # self.error = np.zeros((2, 3))
        self.errorVec = np.zeros((3,))
        self.errorNorm = 0
        self.errorPix = 0
        self.gains_v_kp = np.zeros((3, 1))
        self.gains_v_ki = np.zeros((3, 1))
        self.gains_w_kp = np.zeros((3, 1))
        self.vels = np.zeros((6, 1))
        self.input = np.zeros((6, 1))
        self.Qi = np.array([np.eye(3) for _ in range(n)])

        self.integral = np.zeros((3, 1))
        self.integralTime = time.time()

        self.gain_v_kp = adaptativeGain(
            self.yaml["gain_v_kp_ini"],
            self.yaml["gain_v_kp_max"],
            self.yaml["l_prime_v_kp"],
        )

        self.gain_v_ki = adaptativeGain(
            self.yaml["gain_v_ki_ini"],
            self.yaml["gain_v_ki_max"],
            self.yaml["l_prime_v_ki"],
        )

        self.gain_w_kp = adaptativeGain(
            self.yaml["gain_w_kp_ini"],
            self.yaml["gain_w_kp_max"],
            self.yaml["l_prime_w_kp"],
        )

        self.firstRun = True
        self.smooth = 1

        self.file_input = open(PATH / "out" / f"drone_{drone_id}_input.txt", "w+")
        self.file_error = open(PATH / "out" / f"drone_{drone_id}_error.txt", "w+")
        self.file_errorPix = open(PATH / "out" / f"drone_{drone_id}_errorPix.txt", "w+")
        self.file_time = open(PATH / "out" / f"drone_{drone_id}_time.txt", "w+")
        self.file_v_kp = open(PATH / "out" / f"drone_{drone_id}_v_kp.txt", "w+")
        self.file_v_ki = open(PATH / "out" / f"drone_{drone_id}_v_ki.txt", "w+")
        self.file_w_kp = open(PATH / "out" / f"drone_{drone_id}_w_kp.txt", "w+")
        # self.file_w_ki = open(PATH / "out" / f"drone_{drone_id}_w_ki.txt", "w+")
        self.file_int = open(PATH / "out" / f"drone_{drone_id}_int.txt", "w+")

        # self.save()
        self.Qis = []

    def __name__(self) -> str:
        if self.yaml["control"] == 1:
            return "BearingOnly (-P_gij * gij*)"
        elif self.yaml["control"] == 2:
            return "BearingOnly (gij - gij*)"
        elif self.yaml["control"] == 3:
            return "BearingOnly (-P_gij * gij*) + (gij - gij*)"
        elif self.yaml["control"] == 4:
            return "BearingOnly (-P_gij * gij*) with rotation"
        elif self.yaml["control"] == 5:
            return "BearingOnly (gij - gij*) with rotation"
        elif self.yaml["control"] == 6:
            return "BearingOnly (-P_gij * gij*) + (gij - gij*) with rotation"

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

        self.desiredData.inSphere, self.desiredData.inNormalPlane = sendToSphere(
            self.desiredData.feature, self.yaml["inv_camera_intrinsic_parameters"]
        )
        self.desiredData.bearings = self.middlePoint(self.desiredData.inSphere)
        # self.desiredData.bearings = self.desiredData.inSphere
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

        self.actualData.inSphere, self.actualData.inNormalPlane = sendToSphere(
            self.actualData.feature, self.yaml["inv_camera_intrinsic_parameters"]
        )
        self.actualData.bearings = self.middlePoint(self.actualData.inSphere)
        # self.actualData.bearings = self.actualData.inSphere

        if self.firstRun:
            self.t0L = self.actualTime
            self.tfL = self.t0L + 1

            self.lastQi = np.zeros((len(self.yaml["seguimiento"]), 3, 3))
            self.firstRun = False

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
        self.actualImage = actualImage

        if self.getActualData(actualImage, imgAruco) < 0:
            print("[ERROR] Some ArUco's were not found")
            self.input = np.zeros((6,))
            self.save()
            return self.input

        if self.yaml["control"] in (4, 5, 6):
            print("[INFO] Calculating homographies [Qi]\n")
            if self.homography() < 0:
                print("[ERROR] Homographies could not be calculated")
                self.input = np.zeros((6,))
                self.save()
                return self.input

        U = np.zeros((3, 1))
        Uw = np.zeros((3, 3))
        for index, seg in enumerate(self.yaml["seguimiento"]):
            if self.yaml["control"] == 1:
                temp = (
                    -ortoProj(self.actualData.bearings[index])
                    @ self.desiredData.bearings[index]
                )
            elif self.yaml["control"] == 2:
                temp = (
                    self.actualData.bearings[index] - self.desiredData.bearings[index]
                )
            elif self.yaml["control"] == 3:
                temp = (
                    -ortoProj(self.actualData.bearings[index])
                    @ self.desiredData.bearings[index]
                    + self.actualData.bearings[index]
                    - self.desiredData.bearings[index]
                )
            elif self.yaml["control"] == 4:
                temp = -ortoProj(self.actualData.bearings[index]) @ (
                    (np.eye(3) + self.Qi[index]) / 2 @ self.desiredData.bearings[index]
                )
            elif self.yaml["control"] == 5:
                temp = self.actualData.bearings[index] - (
                    (np.eye(3) + self.Qi[index]) / 2 @ self.desiredData.bearings[index]
                )
            elif self.yaml["control"] == 6:
                temp = (
                    -ortoProj(self.actualData.bearings[index])
                    @ (
                        (np.eye(3) + self.Qi[index])
                        / 2
                        @ self.desiredData.bearings[index]
                    )
                    + self.actualData.bearings[index]
                    - (
                        (np.eye(3) + self.Qi[index])
                        / 2
                        @ self.desiredData.bearings[index]
                    )
                )

            U += temp.reshape(-1, 1)
            if self.yaml["control"] in (4, 5, 6):
                Uw += -(self.Qi[index].T - self.Qi[index])

            print(
                "----------------------------------------",
                f"Actual {seg}   -> {self.actualData.bearings[index]}",
                f"Desired {seg}  -> {self.desiredData.bearings[index]}",
                f"Q{seg}T @ gij  => {self.Qi[index].T @ self.actualData.bearings[index]}",
                f"Q{seg} @ gij   => {self.Qi[index] @ self.actualData.bearings[index]}",
                f"Q{seg}T (deg)  => {np.rad2deg(r2E(self.Qi[index].T))}",
                f"Q{seg} (deg)   => {np.rad2deg(r2E(self.Qi[index]))}",
                "\n",
                sep="\n",
            )

        self.errorVec = np.linalg.norm(U, axis=1).T
        self.errorNorm = np.linalg.norm(self.errorVec)
        self.errorPix = np.linalg.norm(
            self.actualData.inNormalPlane - self.desiredData.inNormalPlane
        )

        self.actualTime = time.time() - self.initTime
        if self.actualTime < self.tfL:
            self.smooth = (
                1 - np.cos(np.pi * (self.actualTime - self.t0L) / (self.tfL - self.t0L))
            ) / 2

        self.gains_v_kp = self.smooth * self.gain_v_kp(2 * self.errorVec)
        self.gains_v_ki = self.smooth * self.gain_v_ki(2 * self.errorVec)
        self.gains_w_kp = self.smooth * self.gain_w_kp(2 * self.errorVec)

        self.integral += np.sign(U) * 0.03245
        for i in range(self.errorVec.shape[0]):
            if self.errorVec[i] < self.yaml["reset_integrator"]:
                self.integral[i] = 0

        self.vels = np.concatenate(
            (
                (self.gains_v_kp * U) + (self.gains_v_ki * self.integral),
                self.gains_w_kp * decomposeSkewSymmetricMatrix(Uw),
            ),
            axis=0,
            dtype=np.float32,
        )

        self.input = np.concatenate(
            (self.rotAndTrans @ self.vels[:3], self.rotAndTrans @ self.vels[3:]),
            axis=0,
            dtype=np.float32,
        ).reshape((6,))

        if self.yaml["control"] == 1:
            self.input[0] *= 2
            # self.input[1] *= 1
            # self.input[2] *= 1
        self.input = np.clip(self.input, -self.yaml["max_vel"], self.yaml["max_vel"])

        if self.yaml["vels"] == 1:
            print("Input before %: ", self.input)
            self.input[:3] = self.input[:3] * 100 / self.yaml["max_vel"]
            self.input = np.clip(self.input, -60, 60)
            self.input[2] = np.clip(self.input[2], -40, 40)
            print("Input  after %: ", self.input)

        self.save()
        return self.input

    def homography(self) -> int:
        self.Qi = []

        ############################################################
        for index, seg in enumerate(self.yaml["seguimiento"]):
            x = normalize(
                np.cross(
                    self.actualData.bearings[index],
                    self.desiredData.bearings[index],
                )
            )
            th = angleVectors(
                self.actualData.bearings[index], self.desiredData.bearings[index]
            )
            X = skewSymmetricMatrix(x)
            R = np.eye(3) + math.sin(th) * X + (1 - math.cos(th)) * (X @ X)
            self.Qi.append(R.T)
            # print(
            #     f"Qi_{index}:\n{r2E(R.T)} - {R.T @ self.actualData.bearings[index]} = {self.desiredData.bearings[index]}"
            # )

        ############################################################
        # for index, seg in enumerate(self.yaml["seguimiento"]):
        #     H = cv2.findHomography(
        #         self.desiredData.feature[4 * index : 4 * index + 4].reshape(-1, 1, 2),
        #         self.actualData.feature[4 * index : 4 * index + 4].reshape(-1, 1, 2),
        #     )[0]

        #     if H is None:
        #         print("[ERROR] Homography could not be calculated")
        #         return -1

        #     # res = (
        #     #     H
        #     #     @ np.hstack(
        #     #         (
        #     #             self.desiredData.feature[4 * index : 4 * index + 4],
        #     #             np.ones((4, 1)),
        #     #         )
        #     #     ).T
        #     # )
        #     # res /= res[2]
        #     # res = np.array(res, dtype=np.int32)

        #     # [
        #     #     print(f"{np.linalg.norm(i-j)}")
        #     #     for i, j in zip(
        #     #         res[0:2, :].T,
        #     #         self.actualData.feature[4 * index : 4 * index + 4],
        #     #     )
        #     # ]

        #     # # H /= np.linalg.norm(H[:, 0])
        #     H /= H[1, 1]

        #     num, Rs, Ts, Ns = cv2.decomposeHomographyMat(
        #         H, self.yaml["camera_intrinsic_parameters"]
        #     )

        #     tries = []
        #     for i in range(num):
        #         if Ns[i][2] > 0:
        #             tries.append([Rs[i].T, Ts[i], Ns[i]])

        #     if len(tries) == 0:
        #         return -1
        #     elif len(tries) == 1:
        #         self.Qi.append(tries[0][0])
        #     else:
        #         if tries[0][2][2] > tries[1][2][2]:
        #             self.Qi.append(tries[1][0])
        #         else:
        #             self.Qi.append(tries[0][0])

        self.Qi = np.array(self.Qi, dtype=np.float32)
        return 0

    def save(self):
        self.actualTime = time.time() - self.initTime
        try:
            self.input.tofile(self.file_input, sep="\t", format="%s")
            self.file_input.write("\n")
            
            self.file_time.write(f"{self.actualTime}\n")
            self.file_errorPix.write(f"{self.errorPix}\n")

            (self.rotAndTrans @ self.errorVec).tofile(
                self.file_error, sep="\t", format="%s"
            )
            self.file_error.write("\n")

            (self.rotAndTrans @ self.gains_v_kp).tofile(
                self.file_v_kp, sep="\t", format="%s"
            )
            self.file_v_kp.write("\n")

            (self.rotAndTrans @ self.gains_v_ki).tofile(
                self.file_v_ki, sep="\t", format="%s"
            )
            self.file_v_ki.write("\n")

            (self.rotAndTrans @ self.gains_w_kp).tofile(
                self.file_w_kp, sep="\t", format="%s"
            )
            self.file_w_kp.write("\n")

            (self.rotAndTrans @ self.integral).tofile(
                self.file_int, sep="\t", format="%s"
            )
            self.file_int.write("\n")

            print(
                f"[INFO]\n",
                f"Error control: {self.errorNorm:.3f}\n",
                f"Error pixels: {self.errorPix:.3f}\n",
                f"Error Vect: {self.errorVec}\n",
                f"Lambda_v_kp -> x: {self.gains_v_kp[1,0]:.3f}, y: {self.gains_v_kp[2,0]:.3f}, z: {self.gains_v_kp[0,0]:.3f}\n",
                f"Lambda_v_ki -> x: {self.gains_v_ki[1,0]:.3f}, y: {self.gains_v_ki[2,0]:.3f}, z: {self.gains_v_ki[0,0]:.3f}\n",
                f"Lambda_w_kp -> x: {self.gains_w_kp[1,0]:.3f}, y: {self.gains_w_kp[2,0]:.3f}, z: {self.gains_w_kp[0,0]:.3f}\n",
            )

        except ValueError as e:
            print("[ERROR] Error writing in file: ", e)

    def middlePoint(self, points: np.ndarray) -> np.ndarray:
        """
        This function returns the middle point of the points in the sphere

        @Params:
          points: np.ndarray -> A (n,3) matrix with the points in the sphere

        @Returns:
          np.ndarray -> A (1,3) matrix with the middle point in the sphere
        """

        temp = []
        for i in range(0, points.shape[0], 4):
            temp.append(normalize(np.mean(points[i : i + 4, :], axis=0)))

        return np.array(temp, dtype=np.float32).reshape(-1, 3)

    def close(self):
        self.file_input.close()
        self.file_time.close()
        self.file_error.close()
        self.file_errorPix.close()
        self.file_v_kp.close()
        self.file_w_kp.close()

        self.file_v_ki.close()
        self.file_int.close()


if __name__ == "__main__":
    img = cv2.imread(f"{PATH}/data/desired_1.jpg")
    control = BearingOnly(img, 1)

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
