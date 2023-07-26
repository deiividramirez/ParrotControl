from math import pi
import numpy as np
import time
import cv2

from pathlib import Path

PATH = Path(__file__).parent.absolute().parent.absolute().parent.absolute()

if __name__ == "__main__":
    # load python file from src/Aux/Funcs.py
    from Funcs import *
else:
    # load python file from Aux/Funcs.py
    from src.Aux.Funcs import *


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

        print(
            f"[INFO] Control law {'(-P_gij * gij*)' if self.yaml['control'] == 1 else '(gij - gij*)'}"
        )

        if self.getDesiredData() < 0:
            print("Desired ArUco not found")
            exit()

        self.storeImage = None
        self.initTime = time.time()
        self.actualTime = time.time()
        self.error = np.zeros((2, 3))
        self.errorVec = np.zeros((1, 3))
        self.errorNorm = 0
        self.errorPix = 0
        self.gains_v_kp = np.zeros((3, 1))
        self.gains_v_ki = np.zeros((3, 1))
        self.gains_w_kp = np.zeros((3, 1))

        self.integral = np.zeros((3, 1))

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

        self.file_int.write("0.0 0.0 0.0\n")
        self.file_v_kp.write("0.0\n")
        self.file_v_ki.write("0.0\n")
        self.file_w_kp.write("0.0\n")
        self.file_time.write("0.0\n")
        self.file_vel_x.write("0.0\n")
        self.file_vel_y.write("0.0\n")
        self.file_vel_z.write("0.0\n")
        self.file_vel_yaw.write("0.0\n")
        self.errorVec.tofile(self.file_error, sep=" ", format="%s")
        self.file_error.write("\n")
        self.file_errorPix.write("0.0\n")

    def __name__(self) -> str:
        return (
            "BearingOnly (-P_gij * gij*)"
            if self.yaml["control"] == 1
            else "BearingOnly (gij - gij*)"
        )

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

        # print("Desired bearings: ", self.desiredData.bearings)
        # exit()

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
        # if np.all(self.storeImage == actualImage):
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

        [
            print(f"Desired -> {i}", f"Actual  -> {j}","\n", sep="\n")
            for i, j in zip(self.desiredData.bearings, self.actualData.bearings)
        ]

        self.error = self.actualData.bearings - self.desiredData.bearings
        self.errorVec = np.linalg.norm(self.error, axis=0)
        self.errorNorm = np.linalg.norm(self.errorVec)
        self.errorPix = np.linalg.norm(
            self.actualData.inNormalPlane - self.desiredData.inNormalPlane
        )

        self.gains_v_kp = np.array(
            [
                self.gain_v_kp(self.errorVec[0], 0),
                self.gain_v_kp(self.errorVec[1], 1),
                self.gain_v_kp(self.errorVec[2], 2),
            ]
        ).reshape(-1, 1)

        self.gains_v_ki = np.array(
            [
                self.gain_v_ki(self.errorVec[0], 0),
                self.gain_v_ki(self.errorVec[1], 1),
                self.gain_v_ki(self.errorVec[2], 2),
            ]
        ).reshape(-1, 1)

        self.gains_w_kp = np.array(
            [
                self.gain_w_kp(self.errorVec[0], 0),
                self.gain_w_kp(self.errorVec[1], 1),
                self.gain_w_kp(self.errorVec[2], 2),
            ]
        ).reshape(-1, 1)

        U = np.zeros((3, 1))
        for i in range(self.actualData.bearings.shape[0]):
            print(-ortoProj(self.actualData.bearings[i]) @ self.desiredData.bearings[i])
            temp = (
                -ortoProj(self.actualData.bearings[i]) @ self.desiredData.bearings[i]
                if self.yaml["control"] == 1
                else self.actualData.bearings[i] - self.desiredData.bearings[i]
            )
            U += temp.reshape(-1, 1)

        self.integral += np.sign(U) * 0.033

        self.vels = np.concatenate(
            (
                (self.gains_v_kp * U) + (self.gains_v_ki * self.integral),
                self.gains_w_kp * np.zeros((3, 1)),
            ),
            axis=0,
            dtype=np.float32,
        )

        self.input = np.concatenate(
            (self.rotAndTrans @ self.vels[:3], self.rotAndTrans @ self.vels[3:]),
            axis=0,
            dtype=np.float32,
        ).reshape((6,))

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
            self.file_errorPix.write(f"{self.errorPix}\n")

            (self.rotAndTrans @ self.errorVec).tofile(self.file_error, sep="\t", format="%s")
            self.file_error.write("\n")

            np.mean(self.gains_v_kp).tofile(self.file_v_kp, sep="\t", format="%s")
            self.file_v_kp.write("\n")

            np.mean(self.gains_v_ki).tofile(self.file_v_ki, sep="\t", format="%s")
            self.file_v_ki.write("\n")

            np.mean(self.gains_w_kp).tofile(self.file_w_kp, sep="\t", format="%s")
            self.file_w_kp.write("\n")

            self.integral.tofile(self.file_int, sep="\t", format="%s")
            self.file_int.write("\n")

            # print(f"x: {self.input[0]}",
            #         f"y: {self.input[1]}",
            #         f"z: {self.input[2]}",
            #         f"yaw: {self.input[5]}",
            #         f"error: {error:.5f}",
            #         f"time: {self.actualTime:.2f}",
            #         sep="\t")

            print(f"[INFO] Error Vect: {self.errorVec}")
            print(f"[INFO] Error: {self.errorNorm}")
        except Exception as e:
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
        self.file_vel_x.close()
        self.file_vel_y.close()
        self.file_vel_z.close()
        self.file_vel_yaw.close()
        self.file_time.close()
        self.file_error.close()
        self.file_errorPix.close()
        self.file_v_kp.close()
        self.file_v_ki.close()
        self.file_w_kp.close()
        # self.file_w_ki.close()
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
