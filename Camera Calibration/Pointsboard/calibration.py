import numpy as np
import cv2
import glob

np.set_printoptions(suppress=True)  # don't use scientific notation

import pathlib

PATH = pathlib.Path(__file__).parent.absolute()

SAMPLES = 30

# Defining the dimensions of circles grid
CIRCLEGRID = (4, 11)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CIRCLEGRID[0] * CIRCLEGRID[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0 : CIRCLEGRID[0], 0 : CIRCLEGRID[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = sorted(glob.glob(f"{PATH}/img/*.jpg"), key=lambda x: int(x.split("/")[-1][:-4]))
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findCirclesGrid(
        gray, CIRCLEGRID, cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
    )
    """
    if ret:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )  # pylint: disable=maybe-no-member
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CIRCLEGRID, corners2, ret)
    """
    cv2.imshow("img", img)
    cv2.waitKey(1)

cv2.destroyAllWindows()

h, w = cv2.imread(images[0]).shape[:2]

# Performing camera calibration by
# passing the value of known 3D points (objpoints)
# and corresponding pixel coordinates of the
# detected corners (imgpoints)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)


# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
# cv_file = cv2.FileStorage(f"{PATH}/out/calibration.yaml", cv2.FILE_STORAGE_WRITE)
# cv_file.write("camera_matrix", mtx)
# cv_file.write("dist_coeff", dist)

# note you *release* you don't close() a FileStorage object
# cv_file.release()

# Print the camera calibration error
error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist
    )  # pylint: disable=maybe-no-member
    error += cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) ** 2 / len(
        imgpoints2
    )

print("Total error: ", error / len(objpoints))
