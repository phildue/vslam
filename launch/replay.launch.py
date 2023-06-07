# Copyright 2022 Philipp.Duernay
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from launch import LaunchDescription

import launch_ros.actions
import launch_ros.descriptions
# TODO put as launch params
FPS = 1
CALIBRATION_PATH = '/home/pi/stereo_ws/config/calibration'
WIDTH = 640
HEIGHT = 480


def generate_launch_description():
    return LaunchDescription([
        launch_ros.actions.Node(
            package='vslam_ros', executable='lukas_kanade_se3_node', output='screen',
            name='lukas_kanade_se3_node')
    ])
