FROM althack/ros2:galactic-full AS developer
# The development environment add build dependencies here
#RUN pip3 install conan && export PATH=$PATH:~/.local/bin/conan
RUN apt update && apt install -y --no-install-recommends libgtk2.0-dev libva-dev libvdpau-dev libboost-python1.71-dev libopencv-dev valgrind kcachegrind libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    libcgal-dev \
    git-lfs \
    clang-format \
    clang-tidy-6.0 \
    ros-galactic-ament-cmake-clang-format \
    ros-galactic-ament-cmake-clang-tidy \
    libfmt-dev
RUN pip3 install opencv-python GitPython sophuspy

# Matplotlib
WORKDIR /opt
RUN git clone https://github.com/lava/matplotlib-cpp.git && cd matplotlib-cpp && mkdir build && cd build && cmake .. && make -j && make install
#RUN git clone https://github.com/artivis/manif.git && cd manif && python3 -m pip install .

# GTest
RUN git clone https://github.com/google/googletest.git && cd googletest && mkdir build && cd build && cmake .. -DBUILD_SHARED_LIBS=ON -DINSTALL_GTEST=ON -DCMAKE_INSTALL_PREFIX:PATH=/usr &&\
make -j2 && make install && ldconfig && cd .. && rm build -r

# Easylogging++
RUN git clone https://github.com/amrayn/easyloggingpp.git && cd easyloggingpp && mkdir build && cd build && cmake .. && make -j2 && make install && cd .. && rm build -r

# Ceres
RUN apt-get install libatlas-base-dev libsuitesparse-dev && git clone https://ceres-solver.googlesource.com/ceres-solver &&\
cd ceres-solver && git checkout $(git describe --tags) && mkdir build && cd build && \
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && make -j2 && make install && cd .. && rm build -r
#RUN git clone https://github.com/phildue/manif-geom-cpp.git && cd manif-geom-cpp && mkdir build && cd build && cmake .. -DBUILD_TESTS=Off && make -j && make install
#RUN git clone https://github.com/goromal/ceres-factors.git && cd ceres-factors && mkdir build && cd build && cmake .. && make && make install
RUN git clone https://github.com/strasdat/Sophus.git && cd Sophus && mkdir build && cd build && cmake .. && make -j2 && make install && cd .. && rm build -r

RUN mkdir -p ros_deps_ws/src && cd ros_deps_ws/src && \
git clone https://github.com/ros-perception/image_pipeline.git && cd image_pipeline && git checkout 457c0c70d9 && \
git clone https://github.com/ros-perception/vision_opencv.git && cd vision_opencv && git checkout 7bbc5ecc232e8 && \
cd /opt/ros_deps_ws/ && colcon build --packages-up-to vision_opencv && echo "source /opt/ros_deps_ws/install/local_setup.bash" >> /home/ros/.bashrc

USER ros
WORKDIR /home/ros/

FROM developer AS builder
# The builder image, additionally contains the source code for compilation
USER ros
ADD --chown=ros:ros . /home/ros/vslam_ros
WORKDIR /home/ros/vslam_ros
RUN colcon build --packages-up-to vslam_ros --parallel-workers 1 --event-handler console_direct+ \
--cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=On \
-DVSLAM_LOG_MINIMAL=On -DVSLAM_LOG_PERFORMANCE_TRACKING=Off -DVSLAM_TEST_VISUALIZE=Off\
-Wall -Wextra -Wpedantic &&\
git rev-parse HEAD > install/COMMIT.sha &&\
touch build/AMENT_IGNORE && touch log/AMENT_IGNORE && touch install/AMENT_IGNORE
RUN echo "source /home/ros/vslam_ros/install/setup.bash" >> /home/ros/.bashrc

FROM developer as runtime
# The final application only copy whats necessary to run
USER ros
COPY --from=builder --chown=ros:ros /home/ros/vslam_ros/install /app/vslam
COPY --from=builder --chown=ros:ros /home/ros/vslam_ros/script /script
RUN pip3 install /script/vslam_evaluation
COPY --from=builder --chown=ros:ros /home/ros/vslam_ros/script/entrypoint.sh /app/vslam/entrypoint.sh
WORKDIR /app/vslam

ENTRYPOINT ["/app/vslam/entrypoint.sh"]
