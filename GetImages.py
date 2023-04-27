from pyparrot.DroneVision import DroneVision
from pyparrot.Bebop import Bebop
from pyparrot.Model import Model

import src.Aux.GUO as GUO

import numpy as np

import threading
import time
import cv2

import pathlib
actualPATH = pathlib.Path(__file__).parent.absolute()

import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file,rtp,udp"

class UserVision:
    def __init__(self, vision: DroneVision, bebot: Bebop, control: classmethod, droneID: int):
        self.index = 0
        self.vision = vision
        self.drone = bebot
        self.status = True
        self.control = control
        self.droneID = droneID

        self.indexImgSave = 0
        self.img = None

        self.drone.set_video_stream_mode("high_reliability")
        self.drone.set_video_framerate("30_FPS")

        # send self.getImages() to thread
        self.actualVision = cv2.VideoCapture(f"{actualPATH}/pyparrot/pyparrot/utils/bebop.sdp")
        print("Sleeping for 5 seconds while getting video stream started up")
        time.sleep(5)

        self.getImagesThread = threading.Thread(target=self.getImages)
        self.saveImagesThread = threading.Thread(target=self.saveImages)
        self.getImagesState = True
        self.getImagesThread.start()
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
                cv2.imshow('Actual image', self.img)
                cv2.waitKey(1)
            else:
                print("Image is None")
    
    def saveImages(self):
        print("Starting thread to save images")
        lastImg = None
        while self.getImagesState:
            if self.img is not None:
                if lastImg is None:
                    lastImg = self.img
                else:
                    if not np.array_equal(lastImg, self.img):
                        lastImg = self.img
                        cv2.imwrite(f"{actualPATH}/img/{self.indexImgSave:05d}.jpg", self.img)
                        self.indexImgSave += 1


    def ImageFunction(self, args):
        
        print("In image function", self.index)
        self.update()
        time.sleep(1)
        # self.drone.smart_sleep(2)
        print("INDEX: ", self.index)
        if self.index == 5:
            print("Closing program")
            self.safe_close()

    def update(self):
        self.index += 1

    def safe_close(self):
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



# Make my bebop object
bebop = Bebop(drone_type="Bebop2")

# Connect to the bebop
success = bebop.connect(5)
bebop.start_video_stream()
bebop.ask_for_state_update()

# set safe indoor parameters
bebop.set_max_tilt(5)
bebop.set_max_vertical_speed(1)

# desiredPath = f"{actualPATH}/data/desired1_f.jpg"
# desiredIMG = cv2.imread(desiredPath)
# R = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
# control1 = GUO.GUO(desiredIMG, 1, R)
# exit()

if success:
    # This function starts the DroneVision class, which contains a thread that starts
    # receiving frames from the bebop.
    bebopVision = DroneVision(bebop, Model.BEBOP)

    # Create the user vision function with the bebopVision object
    userVision = UserVision(bebopVision, bebop, 0, 1)
    
    # Start the vision thread
    bebopVision.set_user_callback_function(userVision.ImageFunction, user_callback_args=None)
    bebopVision._start_video_buffering()

    while userVision.status:
        pass
    
    bebop.disconnect()
else:
    print("Error connecting to bebop. Retry")

