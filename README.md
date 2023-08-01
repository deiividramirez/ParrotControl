# Implementation of an Image Based Visual Servo (IBVS) for a Quadrotor UAV for a follower-leader formation control

This repository was made to implement a modification to the algorithm described in the paper "Image-based estimation, planning, and control for high-speed flying through multiple openings" by Guo, Dejun and Leang for the leaders, and only-bearing based control proposed on "Translational and scaling formation maneuver control via bearing-bases approach" by Shiyu Zhao, Daniel Zelazo. This will be implemented in the phisical drone Parrot Bebop 2 with a python 3 script.



## Let's begin 🚀

The next instructions will allow you to get a copy of the project running on your local machine for development and testing purposes. There are many prerequisites that you need to install before running the code. The code was developed and tested on Ubuntu 20.04 LTS. The code was developed using Python 3.9.16 and ffmpeg version 4.2.7-0ubuntu0.1+esm1.

### Prerequisites 📋

* Python 3.9.16+
* OpenCV 4+
* ffmpeg version 4.2.7-0ubuntu0.1+esm1
* [PyParrot](https://github.com/amymcgovern/pyparrot) (version included in this repository)

### Instalation 🔧


#### Python dependencies

To install all the dependecies in the *'requirements.txt'* file. It is recommended to use a virtual environment to install the dependencies. You can create a virtual environment by running the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then, you can install the dependencies by running the following command:

```bash
pip install -r requirements.txt
```


#### FFMPEG dependency

This project uses ffmpeg just to be able to open the stream video from the drone. To install [ffmpeg](https://ffmpeg.org/), you can run the following command:

```bash
sudo apt install ffmpeg
```



## Pre - Setup for Drone's Flying 🛠️🛸

You can configurate all the parameters in the *"general.yaml"* file, and the necessary parameters for each drone in the *"drone_1.yaml"* file. Both files are inside *"config"* folder.

Explanation of the parameters in the *"general.yaml"* file:

- > **SAVE_IMAGES**: This parameter is useful for saving the images of the drone's camera. If you want to save the images, you have to change the variable to 1.

- > **SAVE_WITH_ARUCO** If you want to save the images with the points given by the ArUco marker, you have to change the variable to 1.

- > **MAX_ITERATIONS**: This parameter is for setting the maximum number of iterations for the control.

- > **MAX_TIME**: This parameter is for setting the maximum time for the control in seconds.

- > **desiredImage** This parameter is the name of the image that will be used for the control. The image must be in the *"/data"* folder.

- > **takeoff**: This parameter is for taking-off the drone or not. This can be helful for testing the control without taking-off the drone, or for saving the desired image.

- > **Leader_Follower** This parameter is for setting the control for the leader or the follower. If you want to control the leader, you have to change the variable to 0, if you want to control the follower, you have to change the variable to 1.

Explanation of the parameters in the *"drone_1.yaml"* file:

- > **camera_intrinsic_parameters** This parameter is for setting the camera intrinsic parameters. The parameters must be in a 3x3 matrix as a vector for the yaml file. You can calibrate the camera drone with the file inside *"/Camera Calibration"* folder. Default: [761.4166, 0, 640, 0, 761.4166, 360, 0, 0, 1]

- > **seguimiento** This parameter is for setting the ArUco marker to keep an eye on. We use the ArUco marker 96, 97, 98, and 99 in our configuration setup.

- > **gain_v_kp_max** This parameter is for setting the maximum value for the proportional gain for the velocity control.

- > **gain_v_kp_ini** This parameter is for setting the initial value for the proportional gain for the velocity control.

- > **l_prime_v_kp** This parameter is for setting the value for the adaptive gain for the velocity control.

- > **gain_v_ki_max** This parameter is for setting the maximum value for the integral gain for the velocity control.

- > **gain_v_ki_ini** This parameter is for setting the initial value for the integral gain for the velocity control.

- > **l_prime_v_ki** This parameter is for setting the value for the adaptive gain for the velocity control.

- > **gain_w_kp_max** This parameter is for setting the maximum value for the proportional gain for the angular velocity control.

- > **gain_w_kp_ini** This parameter is for setting the initial value for the proportional gain for the angular velocity control.

- > **l_prime_w_kp** This parameter is for setting the value for the adaptive gain for the angular velocity control.

- > **max_vel** This parameter is for setting the maximum value for the drone's velocity saturation.

- > **control** This parameter is for setting the control law for the velocity control. For a leader drone, *1* represents the control law *1/dist*, and *2* represents the control law *dist*. For a follower drone, *1* represents the control law *-P_gij * gij*, and *2* represents the control law *gij - gij*.

- > **vels** This parameter is for setting the velocity control in the PyParrot library. *0* represents the *move_relative* command, and *1* represents the *fly_direct* command.

### Disclaimer:

When setting up the parameters, you have to be careful. The parameters 
*l_prime* for any gain must be greater than 0 with the initial value less than the maximum value due to the adaptive gain, which increases when the error approaches to 0. If *l_prime* is less than 0, then the initial value must be greater than the maximum value due to the adaptive gain, which decreases when the error approaches to 0. This is useful for avoiding the chattering effect.

Also, the parameters *gain_v_kp_max*, *gain_v_kp_ini*, *gain_v_ki_max*, *gain_v_ki_ini*, *gain_w_kp_max*, *gain_w_kp_ini* must be greater than 0. If the parameters are less than 0, then the adaptive gain will be negative, and the control will not work properly.


## Python Execution ⚙️

You can run the following command:

```bash
python OpenControl.py
```
To finish the program just make a click in the image window. 

** *DO NOT CLOSE THE TERMINAL IT CAN CAUSE A CRASH IN THE DRONE* **

### Disclaimer ⚠️

This script was developed for the Parrot Bebop 2 drone. The IP address of the drone is discovered by PyParrot library, so it is enough with connecting the drone via wifi to the computer. Due the image proccesing, Python threading, the PyParrot library, and OpenCv library, it is recommended to use one computer for drone. This script was tested in a computer with an Intel Core i5-1035g1, 8GB RAM, and Ubuntu 20.04 LTS. 

## Additional information 📖

* You can find more information about the simulation about this experimental setup in [Github](https://github.com/deiividramirez/bearings).

* If you want to make a mp4 video with the saved images from a previous code execution, you can use the `makeMP4.py` file. The mp4 video will be saved in the *"/out"* folder as *"out_sim_1.mp4"*.


## Autores ✒️

- > **David Ramírez** * [deiividramirez](https://github.com/deiividramirez) *


<!-- ## License 📄 -->

<!-- Este proyecto está bajo la Licencia (Tu Licencia) - mira el archivo [LICENSE.md](LICENSE.md) para detalles -->


---
⌨️ with ❤️ by [deiividramirez](https://github.com/deiividramirez) 😊