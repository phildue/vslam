// Copyright 2022 Philipp.Duernay
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef VSLAM_ODOMETRY_H__
#define VSLAM_ODOMETRY_H__

#include "feature_tracking/FeatureTracking.h"
#include "feature_tracking/FeatureTrackingOcv.h"
#include "feature_tracking/Matcher.h"
#include "feature_tracking/OverlayCorrespondences.h"
#include "feature_tracking/OverlayFeatures.h"
#include "feature_tracking/OverlayMatchCandidates.h"
#include "iterative_closest_point/IterativeClosestPoint.h"
#include "iterative_closest_point/IterativeClosestPointOcv.h"
#include "motion_model/EKFConstantVelocitySE3.h"
#include "motion_model/KalmanFilter.h"
#include "motion_model/MotionModel.h"
#include "rgbd_alignment/RgbdAlignment.h"
#include "rgbd_alignment/RgbdAlignmentOpenCv.h"
#include "rgbd_alignment/SE3Alignment.h"
#endif
