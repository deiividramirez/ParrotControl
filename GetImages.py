from pyparrot.DroneVision import DroneVision
from pyparrot.Bebop import Bebop
from pyparrot.Model import Model


import src.Aux.Funcs as Funcs
import src.Aux.BearingOnly as BO
import src.Aux.GUO as GUO


import numpy as np
import threading
import pathlib
import glob
import time
import sys
import cv2
import os


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file,rtp,udp"
np.set_printoptions(suppress=True, linewidth=sys.maxsize, threshold=sys.maxsize)
actualPATH = pathlib.Path(__file__).parent.absolute()


class successConnection:
    def __init__(self, robot: Bebop, control: classmethod):
        self.control = control
        self.bebop = robot
        self.user = None

    def loop(self):
        # This function starts the DroneVision class, which contains a thread that starts
        # receiving frames from the bebop.
        self.bebopVision = DroneVision(self.bebop, Model.BEBOP)
        print("[INFO] Class DroneVision created")

        # Create the user vision function with the self.bebopVision object
        self.user = UserVision(self.bebopVision, self.bebop, self.control, 1)
        print("[INFO] Class UserVision created")

        # Start the vision thread
        # self.bebopVision.set_user_callback_function(self.user.ImageFunction, user_callback_args=None)
        # self.bebopVision._start_video_buffering()
        # print("Vision thread started")

        # while self.user.status:
        #     self.user.ImageFunction(0)
        self.user.ImageFunction(0)
        self.user.safe_close()


class UserVision:
    def __init__(
        self, vision: DroneVision, bebot: Bebop, control: classmethod, droneID: int
    ):
        self.index = 0
        self.vision = vision
        self.drone = bebot
        self.status = True
        self.control = control
        self.droneID = droneID

        self.indexImgSave = 0
        self.img = None
        self.imgAruco = None
        self.takeImage = False
        self.firstRun = True
        self.thereIsAruco = False

        self.drone.set_video_stream_mode("high_reliability")
        self.drone.set_video_framerate("30_FPS")

        self.yaml = Funcs.loadGeneralYaml(actualPATH)

        self.actualVision = cv2.VideoCapture(
            f"{actualPATH}/pyparrot/pyparrot/utils/bebop.sdp"
        )

        self.clean()

        self.tkinterActive = False
        # self.safetyCheck = threading.Thread(target=self.safety)
        # self.safetyCheck.start()

        self.getImagesThread = threading.Thread(target=self.getImages)
        self.saveImagesThread = threading.Thread(target=self.saveImages)
        self.getImagesState = True
        self.getImagesThread.start()
        if self.yaml["SAVE_IMAGES"]:
            self.saveImagesThread.start()
        self.clicked = False

        # self.drone.safe_takeoff(5)
        # self.drone.smart_sleep(2)
        # print("\n\nBattery: ", self.drone.sensors.battery)
        # print("Flying state: ", self.drone.sensors.flying_state,"\n\n")

    def ImageFunction(self, args):
        try:
            while not self.clicked:
                if self.firstRun:
                    print(
                        "\n[INFO] Sleeping for 5 seconds while getting video stream started up",
                    )

                    for i in range(5):
                        print(
                            f"[INFO] << ** DO NOT CLOSE THE PROGRAM UNTIL SLEEP IS DONE ** >> {5-i} seconds left"
                        )
                        time.sleep(1)

                    self.firstRun = False
                    self.drone.set_max_tilt(5)
                    self.drone.set_max_vertical_speed(1)
                    # self.drone.safe_takeoff(5)
                    # self.drone.ask_for_state_update()

                    print("\n\nBattery: ", self.drone.sensors.battery)
                    print("Flying state: ", self.drone.sensors.flying_state, "\n\n")

                    self.initTime = self.control.initTime = time.time()

                actualTime = time.time() - self.initTime
                if (actualTime) > (self.yaml["MAX_TIME"]) or self.index >= (
                    self.yaml["MAX_ITER"]
                ):
                    cols, rows = os.get_terminal_size(0)
                    print("\n\n\n", "%" * (cols - 2))
                    print("[CLOSING] Closing program", end="")
                    print("\n", "%" * (cols - 2))
                    self.safe_close()
                    break
                else:
                    cols, rows = os.get_terminal_size(0)
                    print("\n", "=" * (cols - 2))
                    print(
                        f'[INFO] Using control: "{self.control.__name__()}" for drone # {self.droneID}\n'
                    )
                    print(
                        f"[INFO] General time: {actualTime:.2f} seconds out of {self.yaml['MAX_TIME']:.2f} seconds"
                    )
                    print(
                        f'[INFO] Iteration {self.index:>6} out of {self.yaml["MAX_ITER"]}\n'
                    )
                    self.update()

                if self.thereIsAruco:
                    self.vels = self.control.getVels(self.img, self.imgAruco)
                else:
                    self.control.input = self.vels = np.zeros((6,))
                    self.control.save()

                print("\n[VELS] ", self.vels)
                print("[INFO] Time in control: ", self.control.actualTime)

                # self.drone.move_relative(
                #     self.vels[0], self.vels[1], self.vels[2], self.vels[5]
                # )

                # print(self.control(self.img, ))
                # self.takeImage = True
                # if self.img is not None:
                #     cv2.imshow('Actual image', self.img)
                #     cv2.waitKey(1)
                # self.drone.smart_sleep(2)
                time.sleep(0.1)
        except Exception as e:
            print(e)
            # self.drone.emergency_land()
            self.safe_close()
        finally:
            print(">> Closing program...")
            self.safe_close()

    def update(self):
        self.index += 1

    def safe_close(self):
        print("\n\n[INFO] Safely closing program and stopping drone...")

        print(">> Closing files...")
        self.control.close()

        print(">> Landing drone...")
        # self.drone.emergency_land()

        print(">> Closing threads...")
        self.status = False
        self.getImagesState = False
        self.vision.vision_running = False

        try:
            self.getImagesThread.join()
        except Exception as e:
            print("Error joining getImagesThread: ", e)
        try:
            self.saveImagesThread.join()
        except Exception as e:
            print("Error joining saveImagesThread: ", e)

        print(">> Closing vision...")
        self.actualVision.release()

        print(">> Closing OpenCV windows...")
        cv2.destroyAllWindows()

        print(">> Disconnecting drone...")
        self.drone.stop_video_stream()
        # self.drone.smart_sleep(2)
        self.drone.disconnect()

        if self.tkinterActive:
            self.root.destroy()
            self.tkinterActive = False

    def getImages(self):
        print("[THREAD] Starting thread to get images")

        cv2.namedWindow("Actual image", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Actual image", self.onMouse)
        cv2.resizeWindow("Actual image", 640, 360)

        while self.getImagesState:
            self.ret, self.img = self.actualVision.read()
            if self.img is not None:
                self.imgAruco = Funcs.get_aruco(self.img, 4)
                if self.imgAruco[1] is not None:
                    cv2.imshow(
                        "Actual image", Funcs.drawArucoPoints(self.img, self.imgAruco)
                    )
                    self.takeImage = True
                    self.thereIsAruco = True
                else:
                    cv2.imshow("Actual image", self.img)
                    self.thereIsAruco = False
                cv2.waitKey(1)

    def saveImages(self):
        print("[THREAD] Starting thread to save images")
        lastImg = None
        while self.getImagesState:
            if self.img is not None:
                if lastImg is None:
                    lastImg = self.img
                else:
                    if not np.array_equal(lastImg, self.img) and self.takeImage:
                        lastImg = self.img
                        cv2.imwrite(
                            f"{actualPATH}/img/{self.indexImgSave:06d}_{self.control.drone_id}.jpg",
                            self.img,
                        )
                        self.indexImgSave += 1
                        self.takeImage = False

    def land(self):
        print("[LANDING] >> EMERGENCY LANDING << ")
        # self.drone.emergency_land()
        self.safe_close()

    def clean(self):
        # delete all images in img folder
        print("[CLEAN] Cleaning img folder")
        files = glob.glob(f"{actualPATH}/img/*.jpg")
        for f in files:
            os.remove(f)

    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.clicked = True
            self.safe_close()


if __name__ == "__main__":
    # Rotation matrix from camera's frame to drone's frame
    R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    YALM = Funcs.loadGeneralYaml(actualPATH)

    if YALM["Leader_Follower"]:
        control = GUO.GUO(cv2.imread(f"{actualPATH}/data/{YALM['desiredImage']}"), 1, R)
    else:
        control = BO.BearingOnly(cv2.imread(f"{actualPATH}/data/{YALM['desiredImage']}"), 1, R)

    # Make my bebop object
    bebop = Bebop(drone_type="Bebop2")

    # Connect to the bebop
    connection = bebop.connect(5)
    bebop.start_video_stream()

    # set safe indoor parameters
    bebop.set_max_tilt(5)
    bebop.set_max_vertical_speed(1)

    if connection:
        # servo = successConnection(bebop, controlGUO)
        servo = successConnection(bebop, control)
        try:
            servo.loop()
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt has been caught.")
            # bebop.safe_land(5)
            servo.user.safe_close()
    else:
        print("Error connecting to bebop. Retry")
