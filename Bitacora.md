# Instalación Bebop-Autonomy

## Instalación OpenCv 3.4.4

~~~bash
cwd=$(pwd)
sudo apt update && sudo apt upgrade -y
cd $cwd

sudo apt -y remove x264 libx264-dev                                                                                        ─╯
 
sudo apt -y install build-essential checkinstall cmake pkg-config yasm
sudo apt -y install git gfortran
sudo apt -y install libjpeg8-dev libjasper-dev libpng12-dev
sudo apt -y install libtiff5-dev
sudo apt -y install libtiff-dev
sudo apt -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
sudo apt -y install libxine2-dev libv4l-dev

cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h
cd $cwd
 
sudo apt -y install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
sudo apt -y install libgtk2.0-dev libtbb-dev qt5-default
sudo apt -y install libatlas-base-dev
sudo apt -y install libfaac-dev libmp3lame-dev libtheora-dev
sudo apt -y install libvorbis-dev libxvidcore-dev
sudo apt -y install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt -y install libavresample-dev
sudo apt -y install x264 v4l-utils

# Optional dependencies
sudo apt -y install libprotobuf-dev protobuf-compiler
sudo apt -y install libgoogle-glog-dev libgflags-dev
sudo apt -y install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
~~~

~~~bash
cvVersion="3.4.4"
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout $cvVersion
cd ..
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout $cvVersion
cd ..
cd opencv
mkdir build
cd build
~~~

### CMake
~~~bash
cmake -D CMAKE_BUILD_TYPE=RELEASE  -D CMAKE_INSTALL_PREFIX=$cwd/installation/OpenCV-"$cvVersion"  -D INSTALL_PYTHON_EXAMPLES=ON  -D WITH_V4L=ON  -D WITH_QT=OFF  -D WITH_OPENGL=ON  -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) -D BUILD_NEW_PYTHON_SUPPORT=ON -D BUILD_opencv_python3=ON -D HAVE_opencv_python3=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules  -D BUILD_EXAMPLES=ON ..

# cmake -D CMAKE_BUILD_TYPE=RELEASE  -D CMAKE_INSTALL_PREFIX=$cwd/installation/OpenCV-"$cvVersion"  -D INSTALL_PYTHON_EXAMPLES=ON  -D WITH_V4L=ON  -D WITH_QT=OFF  -D WITH_OPENGL=ON  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules  -D BUILD_EXAMPLES=ON ..

# -D OPENCV_PYTHON3_INSTALL_PATH=$cwd/OpenCV-$cvVersion-py3/lib/python3.5/site-packages
make -j$(nproc)
sudo make install
~~~

### Para Python
~~~bash
pip install opencv-contrib-python
~~~

## Instalación Bebop-Autonomy

~~~bash
pip install empy catkin_pkg numpy sympy matplotlib future rosdep zeroconf
mkdir -p ~/bebop_ws/src && cd ~/bebop_ws
catkin init
git clone https://github.com/AutonomyLab/bebop_autonomy.git src/bebop_autonomy
rosdep update --rosdistro kinetic
rosdep install --from-paths src -i
~~~

Se debe modificar el archivo  `~/bebop_ws/src/bebop_autonomy/bebop_driver/CMakeLists.txt` cambiando la línea 88: 
~~~
add_dependencies(bebop ${PROJECT_NAME}_gencfg)
~~~
por 
~~~
add_dependencies(bebop ${PROJECT_NAME}_gencfg ${catkin_EXPORTED_TARGETS})
~~~

Después, se debe compilar el workspace y cargar las variables de entorno:

~~~bash
catkin build
source /opt/ros/kinetic/setup.zsh
source ~/bebop_ws/devel/setup.zsh
~~~

Si falla, se debe desinstalar la librería `em` y volver a instalarla con `pip install empy`

Para probar si funciona, se debe conectar el dron por medio de wifi y ejecutar el siguiente comando:

~~~bash
roslaunch bebop_driver bebop_node.launch
~~~

El cual debe ejecutarse sin ningún error y se debe ver la imagen obenida por medio de la cámara del dron en el apartado de `rqt_image_view` de ROS.

## Instalación de PyParrot

~~~bash
cd ~/Downloads
git clone https://github.com/amymcgovern/pyparrot.git
cd pyparrot
python setup.py build
python setup.py install
pip install zeroconf==0.20.0 pyqt5 matplotlib 
sudo apt install ffmpeg
~~~

Se puede probar la instalación por medio del siguiente script. Este script debe ser ejecutado con el dron conectado por medio de wifi y en un espacio abierto.

El dron solo se moverá directo por medio de las velocidades en roll, pitch, yaw y altura. En este ejemplo solo se moverá hacía al frente un metro (tener en cuenta)

~~~python
from pyparrot.Bebop import Bebop

bebop = Bebop(drone_type="Bebop2")

print("connecting")
success = bebop.connect(10)
print(success)

if (success):
    print("Turning on the video")
    bebop.start_video_stream()

    print("Sleeping")
    bebop.smart_sleep(2)

    bebop.ask_for_state_update()

    bebop.safe_takeoff(10)

    # set safe indoor parameters
    bebop.set_max_tilt(5)
    bebop.set_max_vertical_speed(1)

    # trying out the new hull protector parameters - set to 1 for a hull protection and 0 without protection
    #bebop.set_hull_protection(1)

    print("Flying direct: Slow move for indoors")
    bebop.fly_direct(roll=0, pitch=20, yaw=0, vertical_movement=0, duration=2)

    bebop.smart_sleep(5)

    bebop.safe_land(10)

    print("DONE - disconnecting")
    bebop.stop_video_stream()
    bebop.smart_sleep(5)
    print(bebop.sensors.battery)
    bebop.disconnect()
~~~

Para más información: [PyParrot](https://pyparrot.readthedocs.io/en/latest/) o en el [repositorio](https://github.com/amymcgovern/pyparrot)

---
---
27 Marzo:

Al parecer, causaba daño al ejemplo.

~~~bash
pip uninstall opencv-contrib-python
pip install opencv-contrib-python-headless
~~~

Si embargo, aparece el siguiente error `freetype spu text error: Breaking unbreakable line` al parecer, no hay solución para este error.

Apareció otro error, porque no se tenía instalado cv2.imshow() así:

~~~bash
sudo apt-get install libopencv-*
~~~

---
---

Posible solución para leer los archivos de bebop.sdp:

[StackOverflow](https://stackoverflow.com/questions/50591068/how-to-get-the-frame-from-the-vlc-media-player)

==================================================

[[960.41204057   0.         611.25731235]
 [  0.         926.89357784 315.78496314]
 [  0.           0.           1.        ]]
dist (before) : 
[[  -0.26374852   14.43725484   -0.02991786    0.00430483 -315.94588918]]

[[1046.41440426    0.          623.29908693]
 [   0.         1081.77138317  322.62398863]
 [   0.            0.            1.        ]]
dist (before) : 
[[ -0.02305252   8.42386414  -0.04091214  -0.01597508 -38.29620012]]

[[755.49923679   0.         642.60077944]
 [  0.         764.77803338 301.75515872]
 [  0.           0.           1.        ]]
dist (before) : 
[[-0.1825434   0.31850353 -0.02234754  0.01108357  1.3606745 ]]

729.3915812066 0. 627.8429944103
0. 735.3191400119 294.9953987696
0. 0. 100.0000000000
-1.139650275819981962e-01 9.095649772074997086e-02 -1.291885722746134897e-02 7.742207862762149131e-03 -5.667986483598740771e-02

746.89173704 0. 656.09303096
0. 745.03618996 331.04768173
0. 0. 100.00000000
2.683809673826906828e-01 -2.572818668192118352e+00 -5.747139392494016522e-03 -1.508994012435711579e-03 8.642769731208991502e+00

716.9892652007 0. 669.6309499563
0. 718.3055925133 357.8566592113
0. 0. 100.0000000000000000e+00
4.811623520433999102e-01 -6.169050467266804283e+00 -3.438370103795565513e-03 1.751927201971645309e-02 2.237780565436361613e+01

[[749.86691909   0.         646.23196428]
 [  0.         756.15586705 311.99425651]
 [  0.           0.           1.        ]]
dist (before) : 
[[-0.10599919 -0.57198179 -0.01900578  0.01248652  7.54842267]]