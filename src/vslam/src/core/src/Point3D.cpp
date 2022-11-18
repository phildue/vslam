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

//
// Created by phil on 30.06.21.
//
#include <iostream>

#include "Exceptions.h"
#include "Feature2D.h"
#include "Frame.h"
#include "Point3D.h"
namespace pd::vslam
{
std::uint64_t Point3D::_idCtr = 0U;

Point3D::Point3D(const Eigen::Vector3d & position, std::shared_ptr<Feature2D> ft)
: _id(_idCtr++), _position(position)
{
  addFeature(ft);
}

Point3D::Point3D(
  const Eigen::Vector3d & position, const std::vector<std::shared_ptr<Feature2D>> & features)
: _id(_idCtr++), _position(position)
{
  for (const auto & ft : features) {
    addFeature(ft);
  }
}

void Point3D::addFeature(std::shared_ptr<Feature2D> ft) { _features.push_back(ft); }

void Point3D::removeFeatures()
{
  for (auto ft : _features) {
    ft->point() = nullptr;
    if (ft->frame()) {
      ft->frame()->removeFeature(ft);
    }
  }
  _features.clear();
}

void Point3D::removeFeature(std::shared_ptr<Feature2D> ft)
{
  auto it = std::find(_features.begin(), _features.end(), ft);
  if (it == _features.end()) {
    throw pd::Exception(
      "Did not find feature: [" + std::to_string(ft->id()) + " ] in point: [" +
      std::to_string(_id) + "]");
  }

  ft->point() = nullptr;
  _features.erase(it);

  if (ft->frame()) {
    ft->frame()->removeFeature(ft);
  }

  if (_features.size() < 2) {
    removeFeatures();
  }
}
}  // namespace pd::vslam
