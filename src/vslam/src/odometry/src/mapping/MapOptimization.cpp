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

#include "MapOptimization.h"
#include "utils/utils.h"
#define LOG_MAPPING(level) CLOG(level, "mapping")

namespace pd::vslam::mapping
{
MapOptimization::MapOptimization() { Log::get("mapping"); }
void MapOptimization::optimize(
  const std::vector<FrameRgbd::ShPtr> & frames, const std::vector<Point3D::ShPtr> & points) const
{
  BundleAdjustment::UnPtr ba = std::make_unique<BundleAdjustment>();
  for (const auto & p : points) {
    ba->setPoint(p->id(), p->position());
  }
  for (const auto & f : frames) {
    ba->setFrame(f->id(), f->pose().pose(), f->camera()->K());
    for (const auto & ft : f->features()) {
      ba->setObservation(ft->point()->id(), f->id(), ft->position());
    }
  }

  LOG_MAPPING(DEBUG) << "Reprojection error init: " << ba->computeError()
                     << " Per point: " << ba->computeError() / ba->nPoints();

  ba->optimize();

  LOG_MAPPING(DEBUG) << "Reprojection error optimized: " << ba->computeError()
                     << " Per point: " << ba->computeError() / ba->nPoints();

  for (const auto & p : points) {
    p->position() = ba->getPoint(p->id());
  }
  for (const auto & f : frames) {
    f->set(PoseWithCovariance(ba->getPose(f->id()), MatXd::Identity(6, 6)));
  }
}
}  // namespace pd::vslam::mapping
