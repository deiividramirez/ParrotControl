from pyparrot.DroneVision import DroneVision
from pyparrot.Bebop import Bebop
from pyparrot.Model import Model

import src.Aux.Funcs as Funcs
import src.Aux.GUO as GUO

import numpy as np

import threading
import time
import glob
import cv2

import pathlib

actualPATH = pathlib.Path(__file__).parent.absolute()

import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file,rtp,udp"


class successConnection:
    def __init__(self, robot: Bebop, control: classmethod):
        self.control = control
        self.bebop = robot
        self.user = None

    def loop(self):
        # This function starts the DroneVision class, which contains a thread that starts
        # receiving frames from the bebop.
        self.bebopVision = DroneVision(self.bebop, Model.BEBOP)
        print("Class DroneVision created")

        # Create the user vision function with the self.bebopVision object
        self.user = UserVision(self.bebopVision, self.bebop, self.control, 1)
        print("Class UserVision created")

        # Start the vision thread
        # self.bebopVision.set_user_callback_function(self.user.ImageFunction, user_callback_args=None)
        # self.bebopVision._start_video_buffering()
        print("Vision thread started")

        # while self.user.status:
        #     self.user.ImageFunction(0)
        self.user.ImageFunction(0)

        self.bebop.disconnect()


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

        # self.drone.safe_takeoff(5)
        # self.drone.smart_sleep(2)
        # print("\n\nBattery: ", self.drone.sensors.battery)
        # print("Flying state: ", self.drone.sensors.flying_state,"\n\n")

    def getImages(self):
        print("Starting thread to get images")
        while self.getImagesState:
            self.ret, self.img = self.actualVision.read()
            if self.img is not None:
                # self.takeImage = True
                self.imgAruco = Funcs.get_aruco(self.img, 4)
                cv2.namedWindow("Actual image", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Actual image", 640, 360)
                if self.imgAruco[1] is not None:
                    # print(
                    #     f"Aruco detected with ID's {sorted([i for i in self.imgAruco[1][:,0]])}"
                    # )
                    # set window size
                    cv2.imshow(
                        "Actual image", Funcs.drawArucoPoints(self.img, self.imgAruco)
                    )
                    self.takeImage = True
                    self.thereIsAruco = True
                else:
                    cv2.imshow("Actual image", self.img)
                    self.thereIsAruco = False
                cv2.waitKey(1)

    def ImageFunction(self, args):
        try:
            while True:
                if self.firstRun:
                    print(
                        "\nSleeping for 5 seconds while getting video stream started up"
                    )
                    time.sleep(5)

                    self.firstRun = False
                    # self.drone.safe_takeoff(5)
                    self.drone.ask_for_state_update()

                    print("\n\nBattery: ", self.drone.sensors.battery)
                    print("Flying state: ", self.drone.sensors.flying_state, "\n\n")

                    self.initTime = self.control.initTime = time.time()

                print("\nIn image function", self.index)
                self.update()

                if self.thereIsAruco:
                    self.vels = self.control.getVels(self.img, self.imgAruco)
                else:
                    self.vels = [0, 0, 0, 0, 0, 0]
                    
                print("Vels: ", self.vels)
                print("Time in control: ", self.control.actualTime)

                # self.drone.move_relative(
                #     self.vels[0], self.vels[1], self.vels[2], self.vels[5]
                # )

                # print(self.control(self.img, ))
                # self.takeImage = True
                # if self.img is not None:
                #     cv2.imshow('Actual image', self.img)
                #     cv2.waitKey(1)
                # self.drone.smart_sleep(2)
                print("INDEX: ", self.index)
                time.sleep(.1)
                actualTime = time.time() - self.initTime
                if self.index >= 10 and (actualTime) > 10:
                    print("\nClosing program")
                    self.safe_close()
                    break
                else:
                    print(f"Time: {actualTime}")

        except Exception as e:
            print(e)
            # self.drone.safe_land(5)
            self.safe_close()
        finally:
            self.safe_close()

    def update(self):
        self.index += 1

    def safe_close(self):
        print("\nSafely closing program and stopping drone...")

        # self.drone.safe_land(5)

        self.status = False
        self.getImagesState = False
        self.vision.vision_running = False

        self.getImagesThread.join()
        self.actualVision.release()
        cv2.destroyAllWindows()

        self.drone.stop_video_stream()
        self.drone.smart_sleep(2)

        self.drone.disconnect()

        if self.tkinterActive:
            self.root.destroy()
            self.tkinterActive = False

    def saveImages(self):
        print("Starting thread to save images")
        lastImg = None
        while self.getImagesState:
            if self.img is not None:
                if lastImg is None:
                    lastImg = self.img
                else:
                    if not np.array_equal(lastImg, self.img) and self.takeImage:
                        lastImg = self.img
                        cv2.imwrite(
                            f"{actualPATH}/img/{self.indexImgSave:05d}.jpg", self.img
                        )
                        self.indexImgSave += 1
                        self.takeImage = False

    def safety(self):
        # create a tk window with a button to land the drone always on top
        import tkinter as tk

        self.tkinterActive = True

        self.root = tk.Tk()
        self.root.attributes("-topmost", True)
        self.root.title("Bebop2")
        self.root.geometry("300x100")
        self.root.configure(bg="white")

        self.landButton = tk.Button(
            self.root, text="Land", command=self.land, bg="red", fg="white"
        )
        self.landButton.pack(side=tk.LEFT)

        self.root.protocol("WM_DELETE_WINDOW", self.safe_close)

        self.root.mainloop()

    def land(self):
        print("EMERGENCY LANDING")
        # self.drone.emergency_land()
        self.safe_close()

    def clean(self):
        # delete all images in img folder
        print("Cleaning img folder")
        files = glob.glob(f"{actualPATH}/img/*")
        for f in files:
            os.remove(f)


if __name__ == "__main__":
    # Make my bebop object
    bebop = Bebop(drone_type="Bebop2")

    desiredPath = f"{actualPATH}/data/desired_1.jpg"
    desiredIMG = cv2.imread(desiredPath)
    R = np.array([[0, 0, 1], 
                  [1, 0, 0], 
                  [0, 1, 0]])
    controlGUO = GUO.GUO(desiredIMG, 1, R)
    
    # Connect to the bebop
    connection = bebop.connect(5)
    bebop.start_video_stream()

    # set safe indoor parameters
    bebop.set_max_tilt(5)
    bebop.set_max_vertical_speed(1)


    if connection:
        servo = successConnection(bebop, controlGUO)
        try:
            servo.loop()
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt has been caught.")
            # bebop.safe_land(5)
            servo.user.safe_close()
    else:
        print("Error connecting to bebop. Retry")
