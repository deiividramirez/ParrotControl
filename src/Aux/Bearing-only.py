import numpy as np
import cv2

from pathlib import Path
PATH = Path(__file__).parent.absolute().parent.absolute().parent.absolute()

# load python file from src/Aux/Funcs.py
from Funcs import *

class BearingOnly:
    def __init__(self, img_desired: np.ndarray, drone_id: int, RT: np.ndarray = np.eye(3,3)) -> None:
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

        if self.getDesiredData() < 0:
          print("No se encontró el aruco deseado")
          exit()

        
        self.storeImage = None

        self.file_vel_x = open(PATH / "out" / f"drone_{drone_id}_vel_x.txt", "w")
        self.file_vel_y = open(PATH / "out" / f"drone_{drone_id}_vel_y.txt", "w")
        self.file_vel_z = open(PATH / "out" / f"drone_{drone_id}_vel_z.txt", "w")
        self.file_vel_yaw = open(PATH / "out" / f"drone_{drone_id}_vel_yaw.txt", "w")
        self.file_error = open(PATH / "out" / f"drone_{drone_id}_error.txt", "w")
        # self.file_time = open(PATH / "out" / f"drone_{drone_id}_time.txt", "w")
      
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
      temp = get_aruco(self.img_desired_gray)
    
      for index, seg in enumerate(self.yaml["seguimiento"]):
        if seg in temp[1]:
           self.desiredData.feature.append(temp[0][index][0])
        else:
           return -1
      
      # self.desiredData.bearings = self.middlePoint(self.desiredData.feature)
      self.desiredData.bearings = self.yaml["bearings"]
      self.desiredData.feature = np.array(self.desiredData.feature, dtype=np.int32).reshape(-1, 2)

      # # Temporal plot jejeje
      # for i in range(self.desiredData.feature.shape[0]):
      #   cv2.circle(self.img_desired, (self.desiredData.feature[i, 0], self.desiredData.feature[i, 1]), 5, (0, 255,0 ), -1)
      # cv2.circle(self.img_desired, (int(self.desiredData.bearings[0, 0]), int(self.desiredData.bearings[0, 1])), 5, (0, 0, 255), -1)
      # cv2.circle(self.img_desired, (int(self.desiredData.bearings[1, 0]), int(self.desiredData.bearings[1, 1])), 5, (0, 0, 255), -1)

      # cv2.namedWindow("desired", cv2.WINDOW_FULLSCREEN)
      # cv2.imshow("desired", self.img_desired)
      # cv2.waitKey(0)
      # exit()

      self.desiredData.inSphere = sendToSphere(self.desiredData.feature, self.yaml["inv_camera_intrinsic_parameters"])

      return 0
    
    def getActualData(self, actualImage: np.ndarray) -> int:
      """
      This function get the actual data from the actual image, send the points to
      an sphere with the unified model of camera

      @Params:
        actualImage: np.ndarray -> A (n,3) matrix with the actual image

      @Returns: 
        int -> A flag to know if the aruco was found or not
      """
      self.actualData = actualData()
      temp = get_aruco(actualImage)
    
      for index, seg in enumerate(self.yaml["seguimiento"]):
        if seg in temp[1]:
           self.actualData.feature.append(temp[0][index][0])
        else:
           return -1
        
      self.actualData.feature = np.array(self.actualData.feature, dtype=np.int32).reshape(-1, 2)
      
      # # Temporal plot jejeje
      # for i in range(self.actualData.feature.shape[0]):
      #   cv2.circle(self.img_actual, (self.actualData.feature[i, 0], self.actualData.feature[i, 1]), 5, (0, 255,0 ), -1)
      # cv2.circle(self.img_actual, (int(self.actualData.bearings[0, 0]), int(self.actualData.bearings[0, 1])), 5, (0, 0, 255), -1)
      # cv2.circle(self.img_actual, (int(self.actualData.bearings[1, 0]), int(self.actualData.bearings[1, 1])), 5, (0, 0, 255), -1)

      # cv2.namedWindow("actual", cv2.WINDOW_FULLSCREEN)
      # cv2.imshow("actual", self.img_actual)
      # cv2.waitKey(0)
      # exit()

      self.actualData.inSphere = sendToSphere(self.actualData.feature, self.yaml["inv_camera_intrinsic_parameters"])
      self.actualData.bearings = self.middlePoint(self.actualData.inSphere)
      return 0

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
        temp.append(np.mean(points[i:i+4, :], axis=0))

      return np.array(temp, dtype=np.float32).reshape(-1, 3)

    # def laplacianBearing(self, actualBearings: np.ndarray):
    #   """
    #   This function returns the laplacian matrix for the GUO method in the paper "Image-based estimation, planning,
    #   and control for high-speed flying through multiple openings".

    #   @Params:
    #     actualBearing: np.ndarray -> A (n,3) matrix with the actual bearing in the sphere

    #   @Returns:
    #     L: np.ndarray -> A (n,3) matrix with the laplacian matrix
    #   """
    #   n = actualBearings.shape[0]
    #   L = np.zeros((n, 3))

    #   for i in range(n):
    #     temp = - ortoProj(actualBearings[i, :]) @ self.desiredData.bearings[i, :]
    #     L[i, :] = temp

    #   return L

    def getVels(self, actualImage: np.ndarray) -> np.ndarray:
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
      if np.all(self.storeImage == actualImage):
        print("Same image")
        return self.input
      else:
        self.storeImage = actualImage
        
      if self.getActualData(actualImage) == -1:
        return -1
      
      self.error = self.actualData.bearings - self.desiredData.bearings
      # print(self.actualData.bearings)
      # print(self.desiredData.bearings)
      # print(self.error, np.linalg.norm(self.error,ord=1))
      # exit()

      U = np.zeros((3,1))
      for i in range(self.actualData.bearings.shape[0]):
        temp = - ortoProj(self.actualData.bearings[i, :]) @ self.desiredData.bearings[i, :]
        U += temp.reshape(-1, 1)

      self.vels = np.concatenate(
            (U, 
             np.zeros((3, 1))), axis=0)
      
      self.input = np.concatenate(
            (self.rotAndTrans @ self.vels[:3, :],
             self.rotAndTrans @ self.vels[3:, :]), axis=0
      )

      self.file_vel_x.write(f"{self.input[0, 0]}\n")
      self.file_vel_y.write(f"{self.input[1, 0]}\n")
      self.file_vel_z.write(f"{self.input[2, 0]}\n")
      self.file_vel_yaw.write(f"{self.input[3, 0]}\n")

      self.file_error.write(f"{np.linalg.norm(self.error, ord=1)}\n")

      return self.input

    def close(self):
      self.file_vel_x.close()
      self.file_vel_y.close()
      self.file_vel_z.close()
      self.file_vel_yaw.close()
      self.file_error.close()
  
if __name__ == "__main__":
    img = cv2.imread(f"{PATH}/data/desired_1f.jpg")
    bear = BearingOnly(img, 1)

    # print(bear.getDistances(bear.desiredData.inSphere, bear.desiredData.inSphere))
    print(bear.getVels(cv2.imread(f"{PATH}/data/desired_1f.jpg")))
    print(bear.getVels(cv2.imread(f"{PATH}/data/desired_2f.jpg")))
    print(bear.getVels(cv2.imread(f"{PATH}/data/desired_2f.jpg")))
