from pathlib import Path
import numpy as np
import time
import cv2


PATH = Path(__file__).parent.absolute().parent.absolute().parent.absolute()


if __name__ == "__main__":
    # load python file from src/Backend/Funcs.py
    from Funcs import *
else:
    # load python file from Backend/Funcs.py
    from src.Backend.Funcs import *


class GUO:
    def __init__(
        self,
        img_desired: np.ndarray,
        img_desired2: np.ndarray,
        drone_id: int,
        RT: np.ndarray = np.eye(3, 3),
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
            img_desired: np.ndarray -> A (m, n, 3) matrix with the desired image
            img_desired: np.ndarray -> A (m, n, 3) matrix with the desired image
            drone_id: int -> A flag to know which drone is going to be used
            RT: np.ndarray -> A (3,3) matrix with the rotation and translation
                for changing the reference frame from the camera to the drone

        @Returns:
          None

        """
        self.img1 = img_desired
        self.img2 = img_desired2

        self.drone_id = drone_id
        self.rotAndTrans = RT
        self.yaml = load_yaml(PATH, drone_id)

        self.modeChange = False
        self.useKLT = False
        self.firstKLT = True
        self.ORB = cv2.ORB_create(
            nfeatures=1500,
            scoreType=cv2.ORB_FAST_SCORE,
            scaleFactor=1.0,
            nlevels=2,
            edgeThreshold=31,
            patchSize=31,
            fastThreshold=21,
        )
        # self.MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # brute force matcher
        self.MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.followTo = self.yaml["seguimiento"]

        n = len(self.followTo)
        quant = lambda n: (n * 4 - 1) * n * 4 // 2
        if n not in (1, 4):
            raw = input(
                f"[INFO] Using {n} ArUco markers for control law. That is {quant(n)} distances. Continue? (y/n): "
            )
            if raw.lower() != "y":
                exit()
        elif n == 4:
            print("[INFO] Using the 4 most distant points of the 4 ArUco markers")
            n = 1

        print(
            f"[INFO] Control law {'1/dist' if self.yaml['control'] == 1 else 'dist'} with {quant(n)} distances\n"
        )

        self.img_desired = self.img2
        self.img_desired_gray = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        self.followTo = self.yaml["seguimiento2"]
        if self.getDesiredData() < 0:
            print(f"Desired ArUco with IDs {self.followTo} not found for second phase...")
            exit()

        self.img_desired = self.img1
        self.img_desired_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.followTo = self.yaml["seguimiento"]
        if self.getDesiredData() < 0:
            print(f"Desired ArUco with IDs {self.followTo} not found for first phase...")
            exit()

        self.storeImage = None
        self.lastImage = None

        self.initTime = time.time()
        self.actualTime = 0
        self.error = np.zeros((1, 6))
        self.errorNorm = 0
        self.errorPix = 0
        self.vels = np.zeros((6, 1))
        self.input = np.zeros((6, 1))

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

        self.firstRun = True
        self.smooth = 1

        self.file_input = open(PATH / "out" / f"drone_{drone_id}_input.txt", "w+")
        self.file_error = open(PATH / "out" / f"drone_{drone_id}_error.txt", "w+")
        self.file_errorPix = open(PATH / "out" / f"drone_{drone_id}_errorPix.txt", "w+")
        self.file_time = open(PATH / "out" / f"drone_{drone_id}_time.txt", "w+")
        self.file_v_kp = open(PATH / "out" / f"drone_{drone_id}_v_kp.txt", "w+")
        self.file_v_ki = open(PATH / "out" / f"drone_{drone_id}_v_ki.txt", "w+")
        self.file_w_kp = open(PATH / "out" / f"drone_{drone_id}_w_kp.txt", "w+")
        self.file_int = open(PATH / "out" / f"drone_{drone_id}_int.txt", "w+")

        # self.save()

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
        if not self.useKLT:
            temp = get_aruco(self.img_desired_gray, 4)

            for seg in self.followTo:
                if temp[1] is not None and seg in temp[1]:
                    index = np.argwhere(temp[1] == seg)[0][0]
                    self.desiredData.feature.append(temp[0][index][0])
                else:
                    print("ArUco not found")
                    return -1
            self.desiredData.feature = np.array(
                self.desiredData.feature, dtype=np.int32
            ).reshape(-1, 2)

            if len(self.followTo) == 4:
                self.desiredData.feature = np.array(
                    [self.desiredData.feature[i].copy() for i in [1, 4, 10, 13]]
                )

            self.desiredData.inSphere, self.desiredData.inNormalPlane = sendToSphere(
                self.desiredData.feature, self.yaml["inv_camera_intrinsic_parameters"]
            )

            self.desiredData.draw = self.desiredData.feature.copy()

        else:
            # get the points from the image with 200 pixels of left, right, up and down mask for the ORB
            self.mask = np.zeros(self.img_desired_gray.shape, np.uint8)
            self.rectangle = (
                200,
                200,
                self.img_desired_gray.shape[1] - 200,
                self.img_desired_gray.shape[0] - 200,
            )
            self.mask[
                self.rectangle[1] : self.rectangle[3],
                self.rectangle[0] : self.rectangle[2],
            ] = 255

            kp1, des1 = self.ORB.detectAndCompute(self.img_desired, mask=self.mask)
            self.desiredData.keypoints = np.array(
                [kp.pt for kp in kp1], dtype=np.int32
            ).reshape(-1, 2)
            self.desiredData.descriptor = des1

            # # draw the keypoints as points
            # self.desiredData.draw = cv2.drawKeypoints(
            #     self.img_desired,
            #     kp1,
            #     (0, 255, 0),
            #     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            # )

            # cv2.namedWindow("Desired", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("Desired", 640, 480)
            # cv2.imshow("Desired", self.desiredData.draw)
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

        if not self.useKLT:
            for seg in self.followTo:
                if imgAruco[1] is not None and seg in imgAruco[1]:
                    index = np.argwhere(imgAruco[1] == seg)[0][0]
                    self.actualData.feature.append(imgAruco[0][index][0])
                else:
                    # print("ArUco not found")
                    return -1
            self.actualData.feature = np.array(
                self.actualData.feature, dtype=np.int32
            ).reshape(-1, 2)

            if len(self.followTo) == 4:
                self.actualData.feature = np.array(
                    [self.actualData.feature[i] for i in [1, 4, 10, 13]]
                )

            self.actualData.inSphere, self.actualData.inNormalPlane = sendToSphere(
                self.actualData.feature, self.yaml["inv_camera_intrinsic_parameters"]
            )

            self.actualData.draw = self.actualData.feature.copy()
        else:
            kp2, des2 = self.ORB.detectAndCompute(actualImage, mask=self.mask)
            self.actualData.keypoints = np.array(
                [kp.pt for kp in kp2], dtype=np.int32
            ).reshape(-1, 2)
            self.actualData.descriptor = des2

            # # draw the keypoints as points
            # self.actualData.draw = cv2.drawKeypoints(
            #     actualImage,
            #     kp2,
            #     (0, 255, 0),
            #     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            # )

            # cv2.namedWindow("Actual", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("Actual", 640, 480)
            # cv2.imshow("Actual", self.actualData.draw)

            # Match descriptors.
            matches = self.MATCHER.match(
                self.actualData.descriptor, self.desiredData.descriptor
            )

            # # Sort them in the order of their distance.
            # matches = sorted(matches, key=lambda x: x.distance)
            # print("Matches: ", len(matches))
            # # Draw first all the matches.
            # self.actualData.draw = cv2.drawMatches(
            #     actualImage,
            #     kp2,
            #     self.img_desired,
            #     kp2,
            #     matches[:10],
            #     None,
            #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            # )

            # # show matches in 640x480 window
            # cv2.namedWindow("matches", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("matches", 640, 480)
            # cv2.imshow("matches", self.actualData.draw)
            # cv2.waitKey(0)

            # Get the points from the matches
            (
                self.desiredData.feature,
                self.actualData.feature,
            ) = self.getFeatureFromMatches(
                self.desiredData.keypoints, self.actualData.keypoints, matches
            )

            self.actualData.inSphere, self.actualData.inNormalPlane = sendToSphere(
                self.actualData.feature, self.yaml["inv_camera_intrinsic_parameters"]
            )

            self.desiredData.inSphere, self.desiredData.inNormalPlane = sendToSphere(
                self.desiredData.feature, self.yaml["inv_camera_intrinsic_parameters"]
            )

            exit()

        if self.firstRun:
            self.t0L = self.actualTime
            self.tfL = self.t0L + 1

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

        self.distances, self.error = self.getDistances(
            self.actualData.inSphere, self.desiredData.inSphere
        )
        print(" Distances: ", len(self.distances), self.distances)

        self.errorNorm = np.linalg.norm(self.error, ord=1)

        self.L = self.laplacianGUO(self.actualData.inSphere, self.distances)
        self.Lp = np.linalg.pinv(self.L)

        self.actualTime = time.time() - self.initTime
        if self.actualTime < self.tfL:
            self.smooth = (
                1 - np.cos(np.pi * (self.actualTime - self.t0L) / (self.tfL - self.t0L))
            ) / 2

        self.vels = np.concatenate(
            (
                self.gain_v(self.errorNorm) * self.Lp @ self.error,
                self.gain_w(self.errorNorm) * self.rotationControl(),
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

        if self.errorPix < self.yaml["error_threshold"] and not self.modeChange:
            print("[INFO] Changing mode", "=="*25)
            self.changeMode()
            self.modeChange = True
        
        return self.input

    def save(self):
        self.actualTime = time.time() - self.initTime
        try:
            self.input.tofile(self.file_input, sep="\t", format="%s")
            self.file_input.write("\n")

            self.file_time.write(f"{self.actualTime}\n")
            self.file_errorPix.write(f"{self.errorPix}\n")
            self.file_error.write(f"{self.errorNorm}\n")

            self.file_v_kp.write(f"{self.gain_v.gain}\n")
            self.file_w_kp.write(f"{self.gain_w.gain}\n")

            print(
                f"[INFO]\n",
                f"Error control: {self.errorNorm:.3f}\n",
                f"Error pixels: {self.errorPix:.3f}\n",
                f"Lambda_v_kp -> {self.gain_v.gain:3f}\n",
                f"Lambda_w_kp -> {self.gain_w.gain:3f}\n",
            )

        except ValueError as e:
            print("[ERROR] Error writing in file: ", e)

    def changeMode(self):
        # self.useKLT = True
        if not self.modeChange:
            self.img_desired = self.img2
            self.img_desired_gray = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
            self.followTo = self.yaml["seguimiento2"]
            if self.getDesiredData() < 0:
                print(">>>> Points were not refreshed...")
                self.img_desired = self.img1
                self.img_desired_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
                self.followTo = self.yaml["seguimiento"]
                self.getDesiredData()
            else:
                self.modeChange = True
                self.t0L = self.actualTime
                self.tfL = self.t0L + 1
                print(">>>> Mode changed")
            


    def getFeatureFromMatches(
        self, desiredKeypoints: np.ndarray, actualKeypoints: np.ndarray, matches: list
    ) -> tuple:
        """
        This function returns the feature from the matches between the actual and desired images

        @Params:
            desiredKeypoints: np.ndarray -> A (n,2) matrix with the desired keypoints
            actualKeypoints: np.ndarray -> A (n,2) matrix with the actual keypoints
            matches: list -> A list with the matches between the actual and desired keypoints

        @Returns:
            tuple -> A tuple with the desired and actual features (4, 2) matrices

        """
        corners = np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]])
        desiredFeatures = []
        actualFeatures = []
        # print("Matches: ", matches)
        # get the point nearest to the corner
        for corner in corners:
            minDist = 1e9
            dist = [desiredKeypoints[match.queryIdx] - corner for match in matches]
            minDist = np.linalg.norm(dist, axis=1)
            argmin = np.argmin(minDist)

            # while the point is already in the list, get the next nearest point
            while tuple(desiredKeypoints[matches[argmin].queryIdx]) in desiredFeatures:
                minDist[argmin] = 1e9
                argmin = np.argmin(minDist)


            desiredFeatures.append(tuple(desiredKeypoints[matches[argmin].queryIdx]))
            actualFeatures.append(tuple(actualKeypoints[matches[argmin].trainIdx]))

        # draw the keypoints as points
        for i in actualFeatures:
            self.actualData.draw = cv2.circle(
                self.actualImage,
                tuple(i),
                10,
                (0, 255, 0),
                -1,
            )

        for i in desiredFeatures:
            self.desiredData.draw = cv2.circle(
                self.img_desired,
                tuple(i),
                10,
                (0, 255, 0),
                -1,
            )

        cv2.namedWindow("Actual", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Actual", 640, 480)
        cv2.imshow("Actual", self.actualData.draw)

        cv2.namedWindow("Desired", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Desired", 640, 480)
        cv2.imshow("Desired", self.desiredData.draw)
        cv2.waitKey(0)

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
        self.file_input.close()
        self.file_time.close()
        self.file_error.close()
        self.file_errorPix.close()
        self.file_v_kp.close()
        self.file_w_kp.close()


if __name__ == "__main__":
    control = GUO(
        cv2.imread(f"{PATH}/data/desired_1.jpg"),
        cv2.imread(f"{PATH}/data/desired_2.jpg"),
        1,
    )

    print(
        control.getVels(
            cv2.imread(f"{PATH}/data/desired_1.jpg"),
            get_aruco(cv2.imread(f"{PATH}/data/desired_1.jpg"), 4),
        )
    )

    control.changeMode()

    print(
        control.getVels(
            cv2.imread(f"{PATH}/data/desired_2.jpg"),
            get_aruco(cv2.imread(f"{PATH}/data/desired_2.jpg"), 4),
        )
    )

    # Close files
    control.close()
