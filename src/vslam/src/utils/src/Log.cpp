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
// Created by phil on 07.08.21.
//
#include <eigen3/Eigen/Dense>
#include <experimental/filesystem>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/highgui.hpp>
namespace fs = std::experimental::filesystem;
#include "Log.h"

INITIALIZE_EASYLOGGINGPP

namespace pd::vslam
{
std::map<std::string, std::shared_ptr<Log>> Log::_logs = {};
std::map<std::string, std::shared_ptr<LogPlot>> Log::_logsPlot = {};
std::map<std::string, std::shared_ptr<LogImage>> Log::_logsImage = {};
Level Log::_blockLevel = Level::Unknown;
Level Log::_showLevel = Level::Unknown;
std::string LogImage::_rootFolder = "/tmp/log/";
std::shared_ptr<Log> Log::get(const std::string & name)
{
  auto it = _logs.find(name);
  if (it != _logs.end()) {
    return it->second;
  } else {
    _logs[name] = std::make_shared<Log>(name);
  }
  return _logs[name];
}

std::shared_ptr<LogImage> Log::getImageLog(const std::string & name)
{
  auto it = _logsImage.find(name);
  if (it != _logsImage.end()) {
    return it->second;
  } else {
#ifdef ELPP_DISABLE_ALL_LOGS
    _logsImage[name] = std::make_shared<LogImageNull>(name);
#else
    _logsImage[name] = std::make_shared<LogImage>(name);
#endif
    return _logsImage[name];
  }
}
std::shared_ptr<LogPlot> Log::getPlotLog(const std::string & name)
{
  auto it = _logsPlot.find(name);
  if (it != _logsPlot.end()) {
    return it->second;
  } else {
#ifdef ELPP_DISABLE_ALL_LOGS
    _logsPlot[name] = std::make_shared<LogPlotNull>(name);
#else
    _logsPlot[name] = std::make_shared<LogPlot>(name);
#endif
    return _logsPlot[name];
  }
}

Log::Log(const std::string & name) : _name(name)
{
  el::Loggers::getLogger(name);
  el::Configurations defaultConf;
  defaultConf.setToDefault();
  defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
  el::Loggers::reconfigureLogger(name, defaultConf);
}
void Log::configure(const std::string & configFilePath)
{
  el::Configurations config(configFilePath);
  // Actually reconfigure all loggers instead
  el::Loggers::reconfigureLogger(_name, config);
}
std::vector<std::string> Log::registeredLogs()
{
  std::vector<std::string> keys;
  std::transform(Log::_logs.begin(), Log::_logs.end(), std::back_inserter(keys), [&](auto id_l) {
    return id_l.first;
  });
  return keys;
}
std::vector<std::string> Log::registeredLogsImage()
{
  std::vector<std::string> keys;
  std::transform(
    Log::_logsImage.begin(), Log::_logsImage.end(), std::back_inserter(keys),
    [&](auto id_l) { return id_l.first; });
  return keys;
}
std::vector<std::string> Log::registeredLogsPlot()
{
  std::vector<std::string> keys;
  std::transform(
    Log::_logsPlot.begin(), Log::_logsPlot.end(), std::back_inserter(keys),
    [&](auto id_l) { return id_l.first; });
  return keys;
}
LogPlot::LogPlot(const std::string & name, bool block, bool show, bool save)
: _block(block), _show(show), _save(save), _name(name)
{
}
void operator<<(LogPlot::ShPtr log, vis::Plot::ConstShPtr plot) { log->append(plot); }

LogImage::LogImage(const std::string & name, bool block, bool show, bool save)
: _block(block),
  _show(show),
  _save(save),
  _name(name),
  _folder(LogImage::_rootFolder + "/" + name),
  _ctr(0U)
{
}

void LogImage::logMat(const cv::Mat & mat)
{
  if (mat.cols == 0 || mat.rows == 0) {
    throw pd::Exception("Image is empty!");
  }
  if (_show) {
    cv::imshow(_name, mat);
    // cv::waitKey(_blockLevel <= _blockLevelDes ? -1 : 30);
    cv::waitKey(_block ? 0 : 30);
  }
  if (_save) {
    cv::imwrite(_folder + "/" + _name + std::to_string(_ctr++) + ".jpg", mat);
  }
}

void operator<<(LogImage::ShPtr log, vis::Drawable::ConstShPtr drawable) { log->append(drawable); }
}  // namespace pd::vslam
