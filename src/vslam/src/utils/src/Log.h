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

#ifndef VSLAM_LOG_H
#define VSLAM_LOG_H

#include <iostream>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "core/core.h"
#include "easylogging++.h"
#include "visuals.h"

//TODO user should define this
#define SYSTEM(loglevel) CLOG(loglevel, "system")
#define IMAGE_ALIGNMENT(loglevel) CLOG(loglevel, "image_alignment")
#define SOLVER(loglevel) CLOG(loglevel, "solver")

//TODO(me) global variable? why not -.-
#define LOG_ID pd::vslam::Log::id()

#define LOG_IMG(name) pd::vslam::Log::getImageLog(name)
#define LOG_IMG_ID(name, id, image) \
  pd::vslam::Log::getImageLog(name)->append(ImageStamped({image, id}))
#define LOG_MAT_ID(name, id, image) \
  pd::vslam::Log::getImageLog(name)->append(MatdStamped({image, id}))

#define TIMESTAMP pd::vslam::Log::getCurrentTimestamp()
namespace pd::vslam
{
using Level = el::Level;

template <typename T>
struct MatStamped
{
  Eigen::Matrix<T, -1, -1> mat;
  std::string id;
};
typedef MatStamped<image_value_t> ImageStamped;
typedef MatStamped<double> MatdStamped;

class LogImage
{
public:
  typedef std::shared_ptr<LogImage> ShPtr;

  LogImage(const std::string & name, bool block = false, bool show = false, bool save = false);
  void append(const cv::Mat & mat);
  void append(vis::Drawable::ConstShPtr drawable);
  void append(vis::Plot::ConstShPtr drawable);
  void append(vis::Csv::ConstShPtr csv);

  template <typename T>
  void append(const MatStamped<T> & mat)
  {
    if (_show || _save) {
      logMat(vis::drawAsImage(mat.mat.template cast<double>()), mat.id);
    }
  }

  template <typename T>
  void append(const Eigen::Matrix<T, -1, -1> & mat)
  {
    if (_show || _save) {
      logMat(vis::drawAsImage(mat.template cast<double>()));
    }
  }
  bool & block() { return _block; }
  bool & show() { return _show; }
  const bool & save() { return _save; }
  void set(bool show = false, bool block = false, bool save = false, int rate = 1);

  void createDirectories();

  std::string toString() const;
  static std::string & rootFolder() { return _rootFolder; }

protected:
  bool _block;
  bool _show;
  bool _save;
  int _rate;

  static std::string _rootFolder;
  const std::string _name;
  const std::string _folder;

  std::uint64_t _ctr;

  virtual void logMat(const cv::Mat & mat);
  virtual void logMat(const cv::Mat & mat, const std::string & id);
};

class LogImageNull : public LogImage
{
public:
  LogImageNull() : LogImage("") {}
  void logMat(const cv::Mat & UNUSED(plot)) override {}
};

template <typename T>
void operator<<(LogImage::ShPtr log, const Eigen::Matrix<T, -1, -1> & mat)
{
  log->append<T>(mat);
}
void operator<<(LogImage::ShPtr log, vis::Drawable::ConstShPtr drawable);
void operator<<(LogImage::ShPtr log, vis::Plot::ConstShPtr plot);
void operator<<(LogImage::ShPtr log, vis::Csv::ConstShPtr plot);

template <typename T>
void operator<<(LogImage::ShPtr log, MatStamped<T> & mat)
{
  log->append<T>(mat);
}

class Log
{
public:
#ifdef ELPP_DISABLE_ALL_LOGS
  static constexpr bool DISABLED = true;
#else
  static constexpr bool DISABLED = false;
#endif

  static std::shared_ptr<Log> get(const std::string & name);
  static std::shared_ptr<LogImage> getImageLog(const std::string & name);
  static const std::map<std::string, std::shared_ptr<Log>> & loggers() { return _logs; };
  static const std::map<std::string, std::map<Level, std::shared_ptr<LogImage>>> & imageLoggers();
  static std::vector<std::string> registeredLogs();
  static std::vector<std::string> registeredLogsImage();
  static Timestamp getCurrentTimestamp() { return Log::_t; }
  static void setCurrentTimestamp(Timestamp t) { Log::_t = t; }
  static std::string & id() { return _id; }
  Log(const std::string & name);
  void configure(const std::string & configFilePath);

private:
  const std::string _name;
  static std::map<std::string, std::shared_ptr<Log>> _logs;
  static std::map<std::string, std::shared_ptr<LogImage>> _logsImage;
  static Timestamp _t;
  static std::string _id;
};
}  // namespace pd::vslam

#endif  //VSLAM_LOG_H
