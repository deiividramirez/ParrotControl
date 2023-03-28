from pyparrot.DroneVision import DroneVision
from pyparrot.Bebop import Bebop
from pyparrot.Model import Model
import threading
import time
import cv2

class UserVision:
    def __init__(self, vision: DroneVision, bebot: Bebop):
        self.index = 0
        self.vision = vision
        self.drone = bebot
        self.status = True

    def ImageFunction(self, args):
        img = self.vision.get_latest_valid_picture()
        if img is not None:
            # filename = "test_image_%06d.png" % self.index
            # print(f"saving picture on {self.index} {filename}" )
            # cv2.imwrite(filename, img)
            cv2.imshow('Actual image', img)
            cv2.waitKey(1)
            self.index += 1

            if self.index == 500:
                self.status = False



# Make my bebop object
bebop = Bebop(drone_type="Bebop2")

# Connect to the bebop
success = bebop.connect(5)

if success:
    # This function starts the DroneVision class, which contains a thread that starts
    # receiving frames from the bebop.
    bebopVision = DroneVision(bebop, Model.BEBOP)

    # Create the user vision function with the bebopVision object
    userVision = UserVision(bebopVision, bebop)
    
    # Start the vision thread
    bebopVision.set_user_callback_function(userVision.ImageFunction, user_callback_args=None)
    
    # Start the video
    successVideo = bebopVision.open_video()

    # if successVideo:
    #     print("Vision successfully started!")
    #     print("Fly me around by hand!")
    #     bebop.smart_sleep(5)

    #     # bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=-2, duration=4)
    #     # bebop.smart_sleep(25)
    #     print("Finishing demo and stopping vision")

    while userVision.status:
        pass
        # print(userVision.index)
        # if userVision.index == 1:
        # bebop.smart_sleep()
    
    # Stop the vision thread
    bebopVision.close_video()

    # bebop.safe_land(5)


    # disconnect nicely so we don't need a reboot
    bebop.disconnect()
else:
    print("Error connecting to bebop. Retry")

