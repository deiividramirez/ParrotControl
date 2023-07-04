import os

def printInit():
    # get length of screen terminal
    rows, columns = os.popen('stty size', 'r').read().split()

    print(f"""\n\n{'='*int(columns)}
Press the following keys to move the drone:

    w: move *X* forward
    e: move *X* backward
    s: move *Y* left
    d: move *Y* right
    c: move *Z* up
    x: move *Z* down
    v: move *yaw* left
    b: move *yaw* right

    g: less vel
    h: more vel

    t: take off
    l: land
    q: exit the program
    """)
    return input(">> ")

from pyparrot.Bebop import Bebop
from pyparrot.Model import Model

# Make my bebop object
bebop = Bebop(drone_type="Bebop2")

# Connect to the bebop
connection = bebop.connect(5)

# set safe indoor parameters
bebop.set_max_tilt(5)
bebop.set_max_vertical_speed(1)

VEL = .2

while True:
    option = printInit()
    if option == "q":
        exit()
    elif option == "t":
        print("Taking off")
        bebop.safe_takeoff(5)
    elif option == "l":
        print("Landing")
        bebop.safe_land(5)
        print("Landed")
        break
    elif option == "w":
        print("Moving *X* forward")
        bebop.move_relative(VEL, 0, 0, 0)
    elif option == "e":
        print("Moving *X* backward")
        bebop.move_relative(-VEL, 0, 0, 0)
    elif option == "s":
        print("Moving *Y* left")
        bebop.move_relative(0, -VEL, 0, 0)
    elif option == "d":
        print("Moving *Y* right")
        bebop.move_relative(0, VEL, 0, 0)
    elif option == "x":
        print("Moving *Z* down")
        bebop.move_relative(0, 0, VEL, 0)
    elif option == "c":
        print("Moving *Z* up")
        bebop.move_relative(0, 0, -VEL, 0)
    elif option == "v":
        print("Moving *yaw* left")
        bebop.move_relative(0, 0, 0, -.5)
    elif option == "b":
        print("Moving *yaw* right")
        bebop.move_relative(0, 0, 0, .5)
    elif option == "g":
        VEL += .1
        if VEL >= 1:
            VEL = 1
        print("Se subió 0.1")
    elif option == "h":
        if VEL <= 0:
            VEL = 0
        VEL -= .1
        print("Se bajó 0.1")
    else:
        print("Invalid option")
        continue

bebop.disconnect()