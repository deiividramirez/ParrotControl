import numpy as np
import cv2
import glob

np.set_printoptions(suppress=True)  # don't use scientific notation

import pathlib

PATH = pathlib.Path(__file__).parent.absolute()

# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = sorted(
    [f"{i}" for i in glob.glob(f"{PATH}/*.jpg")],
    key=lambda x: int(x.split("/")[-1].split(".")[0]),
)
print(images)

for fname in images:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.waitKey(1)

cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread(images[0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
# make 20 random choices of imgpoints
# to get a better calibration result
mtxMat = []
distMat = []
rvecsMat = []
tvecsMat = []

for i in (lend:=range(30)):
    print(f"Calibrating... {(i+1):02d}/{len(lend):02d}", end="\r")
    # 25 samples of imgpoints into tempimgpoints
    index = np.random.choice(len(imgpoints), 25, replace=False)
    tempimgpoints = [imgpoints[i] for i in index]
    tempobjpoints = [objpoints[i] for i in index]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        tempobjpoints, tempimgpoints, gray.shape[::-1], None, None
    )
    mtxMat.append(mtx)
    distMat.append(dist)
    rvecsMat.append(rvecs)
    tvecsMat.append(tvecs)

mtx = np.mean(mtxMat, axis=0)
dist = np.mean(distMat, axis=0)
rvecs = np.mean(rvecsMat, axis=0)
tvecs = np.mean(tvecsMat, axis=0)

print("Camera matrix (before) : ")
print(mtx)
print("dist (before) : ")
print(dist)

# Try to undistort one of the images
img = cv2.imread(images[0])
h, w = img.shape[:2]
print(f"\nHeight: {h}, Width: {w}\nCx: {w/2}, Cy: {h/2}\n")
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

print("Camera matrix (after) : ")
print(newcameramtx)
print("dist (after) : ")
print(roi)

np.savetxt(f'{PATH}/cameraMatrix.txt', newcameramtx)
np.savetxt(f'{PATH}/distCoeffs.txt', dist)

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y : y + h, x : x + w]
cv2.imwrite(f"{PATH}/calibresult.png", dst)


# Reprojection error
mean_error = 0
for i in range(len(rvecs)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Total error: ", mean_error / len(objpoints))
