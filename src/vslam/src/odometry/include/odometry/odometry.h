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

#include "KeyFrameSelection.h"
#include "MotionPrediction.h"
#include "Odometry.h"
#include "direct_image_alignment/RgbdAlignmentOpenCv.h"
#include "direct_image_alignment/SE3Alignment.h"
#include "feature_tracking/FeatureTracking.h"
#include "feature_tracking/Matcher.h"
#include "iterative_closest_point/IterativeClosestPoint.h"
#include "iterative_closest_point/IterativeClosestPointOcv.h"
#include "mapping/BundleAdjustment.h"
#include "mapping/Map.h"
#endif
