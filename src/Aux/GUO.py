import numpy as np
import yaml
import cv2

from pathlib import Path
PATH = Path(__file__).parent.absolute().parent.absolute().parent.absolute()

# load python file from src/Aux/Funcs.py
from Funcs import *

class GUO:
    def __init__(self, img_desired: np.ndarray, drone_id: int) -> None:
        self.img_desired = img_desired
        self.img_desired_gray = cv2.cvtColor(img_desired, cv2.COLOR_BGR2GRAY)
        self.drone_id = drone_id

        self.yaml = load_yaml(PATH, drone_id)

        if self.getDesiredData() < 0:
          print("No se encontró el aruco deseado")
          exit()
      
    def getDesiredData(self) -> int:
      self.desiredData = desiredData()
      temp = get_aruco(self.img_desired_gray)
    
      for index, seg in enumerate(self.yaml["seguimiento"]):
        if seg in temp[1]:
           self.desiredData.feature.append(temp[0][index][0])
        else:
           return -1
      self.desiredData.feature = np.array(self.desiredData.feature, dtype=np.int32).reshape(-1, 2)

      # Temporal plot jejeje
      # for i in range(self.desiredData.feature.shape[0]):
      #   cv2.circle(self.img_desired, (self.desiredData.feature[i, 0], self.desiredData.feature[i, 1]), 5, (0, 255,0 ), -1)
      
      # cv2.imshow("desired", self.img_desired)
      # cv2.waitKey(0)

      self.desiredData.inSphere = sendToSphere(self.desiredData.feature, self.yaml["inv_camera_intrinsic_parameters"])
      
      return 0
    
    def getActualData(self, actualImage: np.ndarray) -> int:
      self.actualData = actualData()
      temp = get_aruco(actualImage)
    
      for index, seg in enumerate(self.yaml["seguimiento"]):
        if seg in temp[1]:
           self.actualData.feature.append(temp[0][index][0])
        else:
           return -1
      self.actualData.feature = np.array(self.actualData.feature, dtype=np.int32).reshape(-1, 2)

      # Temporal plot jejeje
      # for i in range(self.actualData.feature.shape[0]):
      #   cv2.circle(self.img_actual, (self.actualData.feature[i, 0], self.actualData.feature[i, 1]), 5, (0, 255,0 ), -1)
      
      # cv2.imshow("actual", self.img_actual)
      # cv2.waitKey(0)

      self.actualData.inSphere = sendToSphere(self.actualData.feature, self.yaml["inv_camera_intrinsic_parameters"])
      
      return 0
      
    def getDistances(self, pointsActual: np.ndarray, pointsDesired: np.ndarray, CONTROL: int = 1) -> tuple:
      distances, error = [], []
      for i in range(pointsActual.shape[0]):
          for j in range(i):
          # for j in range(pointsActual.shape[1]):
              if i != j:
                  dist  = np.sqrt( 2 - 2 * np.dot(pointsDesired[i], pointsDesired[j]) )
                  dist2 = np.sqrt( 2 - 2 * np.dot(pointsActual[i], pointsActual[j]) ) 
                  if dist <= 1e-9 or dist2 <= 1e-9:
                      continue

                  distances.append(
                      dictDist(i, j, 1/dist, 1/dist2) if CONTROL == 1 else dictDist(i, j, dist, dist2)
                  )
              
      distances = sorted(distances, key=lambda x: x.dist, reverse=True)
      error = [distance.dist2 - distance.dist for distance in distances]
      return distances, np.array(error).reshape(len(error), 1)

    def laplacianGUO(self, pointsSphere: np.ndarray, distances: dictDist, CONTROL:int = 1):
      n = len(distances)
      L = np.zeros((n, 3))

      for i in range(n):
          s = -distances[i].dist**3 if CONTROL == 1 else 1/distances[i].dist
          
          temp = s * ( (pointsSphere[distances[i].i].reshape(1,3)) @ ortoProj(pointsSphere[distances[i].j].reshape(3,1)) + 
                      (pointsSphere[distances[i].j].reshape(1,3)) @ ortoProj(pointsSphere[distances[i].i].reshape(3,1)) )
          
          L[i, :] = temp
      return L

    def getVels(self, actualImage: np.ndarray) -> np.ndarray:

      if self.getActualData(actualImage) == -1:
        return -1
      
      self.distances, self.error = self.getDistances(self.actualData.inSphere, self.desiredData.inSphere)
      self.L = self.laplacianGUO(self.actualData.inSphere, self.distances)
      self.Lp = np.linalg.pinv(self.L)
      self.vels = np.concatenate(
            (self.Lp @ self.error, 
             np.zeros((3, 1))), axis=0)
      return self.vels
        
  
if __name__ == "__main__":
    img = cv2.imread(f"{PATH}/data/desired_1f.jpg")
    guo = GUO(img, 1)

    # print(guo.getDistances(guo.desiredData.inSphere, guo.desiredData.inSphere))
    print(guo.getVels(cv2.imread(f"{PATH}/data/desired_1f.jpg")))