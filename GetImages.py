# MAIN PYPARROT LIBRARY FOR DRONE CONTROL
from pyparrot.DroneVision import DroneVision
from pyparrot.Bebop import Bebop
from pyparrot.Model import Model

# IMPORTS FROM FILES
import src.Aux.Funcs as Funcs
import src.Aux.BearingOnly as BO
import src.Aux.GUO as GUO

# MAIN LIBRARIES
import numpy as np
import threading
import pathlib
import glob
import time
import sys
import cv2
import os

# COLORAMA LIBRARY FOR COLORFUL PRINTS
from colorama import init as colorama_init, Fore, Style

# OPENCV CONECTION WITH FFMPEG
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file,rtp,udp"
# NUMPY CONFIGURATION
np.set_printoptions(suppress=True, linewidth=sys.maxsize, threshold=sys.maxsize)
# ACTUAL PATH CONFIGURATION
actualPATH = pathlib.Path(__file__).parent.absolute()


class successConnection:
    def __init__(self, robot: Bebop, control: (GUO.GUO and BO.BearingOnly)):
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

        self.user.ImageFunction(0)

        print("[INFO] Closing from 'successConnection' class")
        self.user.safe_close()


class UserVision:
    def __init__(
        self, vision: DroneVision, bebot: Bebop, control: classmethod, droneID: int
    ):
        self.index = 0
        self.indexImgSave = 0

        self.vision = vision
        self.drone = bebot
        self.control = control
        self.droneID = droneID

        self.img = None
        self.imgAruco = None

        self.status = True
        self.firstRun = True

        self.takeImage = False
        self.thereIsAruco = False
        self.clicked = False

        self.drone.set_video_stream_mode("high_reliability")
        self.drone.set_video_framerate("30_FPS")
        self.drone.enable_geofence(0)

        self.yaml = Funcs.loadGeneralYaml(actualPATH)

        self.actualVision = cv2.VideoCapture(
            f"{actualPATH}/pyparrot/pyparrot/utils/bebop.sdp"
        )

        self.getImagesThread = threading.Thread(target=self.getImages)
        self.saveImagesThread = threading.Thread(target=self.saveImages)

        self.clean()
        self.getImagesState = True
        self.getImagesThread.start()
        if self.yaml["SAVE_IMAGES"]:
            self.saveImagesThread.start()

        print(f"{Fore.GREEN}[INFO] Class UserVision created{Style.RESET_ALL}")

    def ImageFunction(self, args):
        cols, rows = os.get_terminal_size(0)

        try:
            print(
                f"\n{Fore.YELLOW}{'*' * (cols - 4)}{Style.RESET_ALL}\n[INFO] Asking for state update"
            )
            self.drone.ask_for_state_update()

            while not self.clicked:
                if self.drone.sensors.flying_state == "emergency":
                    print(
                        f"\n\n\n{Fore.RED}[EMERGENCY] ** Emergency landing **{Style.RESET_ALL}"
                    )
                    if self.drone.safe_land(5) < 0:
                        print(f"{Fore.RED}[ERROR] Error landing drone{Style.RESET_ALL}")
                        self.drone.emergency_land()
                    else:
                        print(f"{Fore.GREEN}[INFO] Drone landed{Style.RESET_ALL}")
                    self.safe_close()
                    break

                if self.firstRun:
                    print(
                        "\n[INFO] Sleeping for 5 seconds while getting video stream started up",
                    )

                    for i in range(5):
                        print(f"[INFO] >> {5-i} seconds left", end="\r")
                        if self.clicked:
                            break
                        time.sleep(1)

                    self.firstRun = False
                    self.drone.set_max_tilt(5)
                    self.drone.set_max_vertical_speed(1)

                    print(f"{Fore.GREEN}[INFO] Safe takeoff{Style.RESET_ALL}")
                    # self.drone.safe_takeoff(5)

                    print("\n\nBattery: ", self.drone.sensors.battery)
                    print("Flying state: ", self.drone.sensors.flying_state, "\n\n")

                    self.initTime = self.control.initTime = time.time()

                    if self.clicked:
                        break

                actualTime = time.time() - self.initTime
                if (
                    actualTime > (self.yaml["MAX_TIME"])
                    or self.index >= self.yaml["MAX_ITER"]
                ):
                    print(f"\n\n\n{Fore.YELLOW}", "%" * (cols - 4), Style.RESET_ALL)
                    print("[CLOSING] Closing program", end="")
                    print(f"\n{Fore.YELLOW}", "%" * (cols - 4), Style.RESET_ALL)
                    self.safe_close()
                    break
                else:
                    print(f"\n{Fore.YELLOW}", "=" * (cols - 4), Style.RESET_ALL)
                    print(
                        f'\t[INFO] Using control: {Fore.RED}"{self.control.__name__()}"{Style.RESET_ALL} for drone # {self.droneID}\n',
                        f'\t[INFO] General time: {Fore.RED}{actualTime:.2f}{Style.RESET_ALL} seconds out of {Fore.RED}{self.yaml["MAX_TIME"]:.2f}{Style.RESET_ALL} seconds\n',
                        f'\t[INFO] Iteration {Fore.RED}{self.index:>6}{Style.RESET_ALL} out of {Fore.RED}{self.yaml["MAX_ITER"]}{Style.RESET_ALL} iterations\n',
                    )
                    self.update()

                if self.thereIsAruco:
                    self.vels = self.control.getVels(self.img, self.imgAruco)
                else:
                    print("[INFO] No aruco detected")
                    self.control.input = self.vels = np.zeros((6,))
                    self.control.save()

                print(f"\n\t[VELS] {self.vels}")
                print(f"\t[INFO] Time in control: {self.control.actualTime:.2f}")

                # self.drone.move_relative(
                #     self.vels[0], self.vels[1], self.vels[2], self.vels[5]
                # )

                time.sleep(0.1)
        except Exception as e:
            print(f"{Fore.RED}[ERROR] {e}\n>> Closing program...{Style.RESET_ALL}")
            # self.drone.emergency_land()
            self.safe_close()
        finally:
            print(">> Finally -> Closing program...")
            self.safe_close()

    def update(self):
        self.index += 1

    def safe_close(self):
        print("\n\n[INFO] Safely closing program and stopping drone...")
        self.clicked = True

        print(">> Closing files...")
        self.control.close()

        print(
            f">> Landing drone... ",
            f"Actual state of drone: {Fore.RED}{self.drone.sensors.flying_state}{Style.RESET_ALL}",
        )
        if self.drone.safe_land(5) < 0:
            print(f"{Fore.RED}[ERROR] Error landing drone{Style.RESET_ALL}")
            self.drone.emergency_land()
        else:
            print(f"{Fore.GREEN}[INFO] Drone landed{Style.RESET_ALL}")

        print(">> Closing threads...")
        self.status = False
        self.getImagesState = False
        self.vision.vision_running = False

        try:
            self.getImagesThread.join()
        except Exception as e:
            print(
                f"{Fore.RED}[ERROR] Error joining func 'getImagesThread': e -> {e}{Style.RESET_ALL}"
            )
        try:
            self.saveImagesThread.join()
        except Exception as e:
            print(
                f"{Fore.RED}[ERROR] Error joining func 'saveImagesThread': e -> {e}{Style.RESET_ALL}"
            )

        print(">> Closing OpenCV windows...")
        cv2.destroyAllWindows()

        print(">> Closing vision...")
        self.actualVision.release()
        self.drone.stop_video_stream()

        print(">> Disconnecting drone...")
        # self.drone.smart_sleep(2)
        self.drone.disconnect()
        self.drone.drone_connection.disconnect()

        print(f"{Fore.CYAN}[INFO] safe_close() finished\n{Style.RESET_ALL}")

    def getImages(self):
        print(f"{Fore.GREEN}[THREAD] Starting thread to get images{Style.RESET_ALL}")

        cv2.namedWindow("Actual image", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Actual image", self.onMouse)
        cv2.resizeWindow("Actual image", 640, 360)

        while self.getImagesState and not self.clicked:
            try:
                self.ret, self.img = self.actualVision.read()
            except Exception as e:
                print(
                    f"{Fore.RED}[ERROR] Error getting image from Bebop2: e -> {e}{Style.RESET_ALL}"
                )

            if self.img is not None and not self.clicked:
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

        cv2.destroyAllWindows()

    def saveImages(self):
        print(f"{Fore.GREEN}[THREAD] Starting thread to save images{Style.RESET_ALL}")
        lastImg = None
        while self.getImagesState and not self.clicked:
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
        if event == cv2.EVENT_LBUTTONUP and not self.clicked:
            cols, rows = os.get_terminal_size(0)
            print(
                f"\n\n{Fore.RED}{'=' * (cols - 4)}\n{Style.RESET_ALL}\n",
                f"{Fore.GREEN}[CLICK] Clicked on screen. Stopping drone #{self.droneID}{Style.RESET_ALL}",
            )
            self.clicked = True
            self.safe_close()


if __name__ == "__main__":
    # Rotation matrix from camera's frame to drone's frame
    R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    YALM = Funcs.loadGeneralYaml(actualPATH)

    if YALM["Leader_Follower"]:
        control = GUO.GUO(cv2.imread(f"{actualPATH}/data/{YALM['desiredImage']}"), 1, R)
    else:
        control = BO.BearingOnly(
            cv2.imread(f"{actualPATH}/data/{YALM['desiredImage']}"), 1, R
        )

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
