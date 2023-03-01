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
#include <fmt/core.h>
using fmt::format;
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
std::map<std::string, std::shared_ptr<LogImage>> Log::_logsImage = {};
std::string LogImage::_rootFolder = "/tmp/log/";
Timestamp Log::_t = 0UL;
std::string Log::_id = "";

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
  } else if (Log::DISABLED) {
    _logsImage[name] = std::make_shared<LogImageNull>();
  } else {
    _logsImage[name] = std::make_shared<LogImage>(name);
  }
  return _logsImage[name];
}

Log::Log(const std::string & name) : _name(name)
{
  el::Loggers::getLogger(name);
  el::Configurations defaultConf;
  defaultConf.setToDefault();
  defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "%datetime %level %msg");
  defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
  el::Loggers::reconfigureLogger(name, defaultConf);
}
void Log::configure(const std::string & configFilePath)
{
  el::Configurations config(configFilePath);
  // Actually reconfigure all loggers instead
  el::Loggers::reconfigureLogger(_name, config);
}
std::string LogImage::toString() const
{
  std::stringstream ss;
  ss << "[" << _name << "]"
     << "\nBlock: [" << (_block ? "On" : "Off") << "]\nShow: [" << (_show ? "On" : "Off")
     << "]\nSave: [" << (_save ? "On" : "Off") << "] to [" << rootFolder() << "/" << _folder << "]";
  return ss.str();
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

void LogImage::append(vis::Plot::ConstShPtr plot)
{
  if (_ctr++ % _rate != 0) {
    return;
  }
  if (_show) {
    plot->plot();
    vis::plt::show(_block);
  }
  if (_save) {
    std::stringstream ss;
    //ss << format("{}/{}/{}_{}.jpg", rootFolder(), _folder, _name, _ctr);
    //vis::plt::save(ss.str());
    std::string filename = format("{}/{}/{}_{}.csv", rootFolder(), _folder, _name, plot->id());
    std::fstream fout;
    fout.open(filename, std::ios_base::out);

    if (!fout.is_open()) {
      throw std::runtime_error("Could not open file at: " + filename);
    }
    fout << plot->csv();
    fout.close();
  }
}
void LogImage::append(vis::Csv::ConstShPtr csv)
{
  if (_ctr++ % _rate != 0) {
    return;
  }
  if (_show || _save) {
    if (_show) {
      std::cout << csv->csv() << std::endl;
    }
    if (_save) {
      std::stringstream ss;
      std::string filename = format("{}/{}/{}_{}.csv", rootFolder(), _folder, _name, csv->id());
      std::fstream fout;
      fout.open(filename, std::ios_base::out);

      if (!fout.is_open()) {
        throw std::runtime_error("Could not open file at: " + filename);
      }
      fout << csv->csv();
      fout.close();
    }
  }
}
void LogImage::append(const cv::Mat & mat)
{
  if (_show || _save) {
    logMat(mat);
  }
}
void LogImage::append(vis::Drawable::ConstShPtr drawable)
{
  if (_show || _save) {
    logMat(drawable->draw());
  }
}

LogImage::LogImage(const std::string & name, bool block, bool show, bool save)
: _block(block), _show(show), _save(save), _rate(1), _name(name), _folder(name), _ctr(0U)
{
  if (_save) {
    createDirectories();
  }
}
void LogImage::createDirectories()
{
  if (!fs::exists(_folder)) {
    fs::create_directories(rootFolder() + "/" + _folder);
  }
}

void LogImage::set(bool show, bool block, bool save, int rate)
{
  _block = block;
  _show = show;
  _save = save;
  _rate = rate <= 0 ? 1 : rate;
  if (_save) {
    createDirectories();
  }
}

void LogImage::logMat(const cv::Mat & mat)
{
  if (_ctr++ % _rate != 0) {
    return;
  }
  if (mat.cols == 0 || mat.rows == 0) {
    throw pd::Exception("Image is empty!");
  }

  if (_show) {
    cv::imshow(_name, mat);
    // cv::waitKey(_blockLevel <= _blockLevelDes ? -1 : 30);
    cv::waitKey(_block ? 0 : 30);
  }
  if (_save) {
    std::stringstream ss;
    ss << format("{}/{}/{}_{}.jpg", rootFolder(), _folder, _name, _ctr);
    cv::imwrite(ss.str(), mat);
  }
}

void LogImage::logMat(const cv::Mat & mat, const std::string & id)
{
  if (_ctr++ % _rate != 0) {
    return;
  }
  if (mat.cols == 0 || mat.rows == 0) {
    throw pd::Exception("Image is empty!");
  }

  if (_show) {
    cv::imshow(_name, mat);
    // cv::waitKey(_blockLevel <= _blockLevelDes ? -1 : 30);
    cv::waitKey(_block ? 0 : 30);
  }
  if (_save) {
    std::stringstream ss;
    ss << format("{}/{}/{}_{}.jpg", rootFolder(), _folder, _name, id);
    cv::imwrite(ss.str(), mat);
  }
}

void operator<<(LogImage::ShPtr log, vis::Drawable::ConstShPtr drawable) { log->append(drawable); }
void operator<<(LogImage::ShPtr log, vis::Plot::ConstShPtr plot) { log->append(plot); }
void operator<<(LogImage::ShPtr log, vis::Csv::ConstShPtr plot) { log->append(plot); }

}  // namespace pd::vslam
