//
// Created by phil on 08.08.21.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>
#include <opencv4/opencv2/core/eigen.hpp>
#include "utils/utils.h"
#include "Exceptions.h"
#include <filesystem>
namespace fs = std::filesystem;
namespace pd::vslam{
        

        void utils::throw_if_nan(const Eigen::MatrixXd &mat, const std::string &msg)
        {
            auto result = mat.norm();
            if (std::isnan(mat.norm()) || std::isinf(result))
            {
                std::stringstream ss;
                ss << mat;
                throw pd::Exception(msg + " contains nan: \n" + ss.str());
            }
        }

        Image utils::loadImage(const fs::path& path, int height, int width, bool grayscale)
        {
            auto mat = cv::imread(path.string(), grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);

            if (mat.cols == 0 || mat.rows == 0)
            {
                throw pd::Exception("Failure during load of: " + path.string());
            }

            if ( height > 0 && width > 0)
            {
                cv::resize(mat,mat,cv::Size(width,height));
            }

            Image img;
            cv::cv2eigen(mat,img);
            return img;
        }

        Eigen::MatrixXd utils::loadDepth(const fs::path& path, int height, int width )
        {
            auto mat = cv::imread(path.string(), cv::IMREAD_ANYDEPTH);

            if (mat.cols == 0 || mat.rows == 0)
            {
                throw pd::Exception("Failure during load of: " + path.string());
            }

            if ( height > 0 && width > 0)
            {
                cv::resize(mat,mat,cv::Size(width,height));
            }

            Eigen::MatrixXd img;
            cv::cv2eigen(mat,img);
            return img.array().isNaN().select(0,img);

        }
        std::map<Timestamp,SE3d> utils::loadTrajectory(const fs::path& path)
        {
            std::ifstream gtFile;
            gtFile.open(path.string());
            
            if(!gtFile.is_open()) {  std::runtime_error("Could not open file at: " + path.string()); }

            std::map<Timestamp,SE3d> poses;
            std::string line;
            while(getline(gtFile, line)) {
                    
                std::vector<std::string> elements;
                std::string s;
                std::istringstream lines(line);    
                while (getline(lines, s, ' ')) { elements.push_back(s); }

                if(elements[0] == "#") {continue; }//Skip comments

                Eigen::Vector3d trans;
                trans << std::stod(elements[1]),std::stod(elements[2]),std::stod(elements[3]);
                Eigen::Quaterniond q(std::stod(elements[7]),std::stod(elements[4]),std::stod(elements[5]),std::stod(elements[6]));
                std::vector<std::string> tElements;
                std::string st;
                std::istringstream tLine(elements[0]);    
                while (getline(tLine, st, '.')) { tElements.push_back(st); }

                auto sec = std::stoull(tElements[0]);
                auto nanosec = std::stoull(tElements[1])*100000;
                poses.insert({sec*1e9+nanosec,SE3d (q,trans)});
            }
            return poses;
        }


        void utils::saveImage(const Image& img,const fs::path& path)
        {
            cv::Mat mat;
            cv::eigen2cv(img,mat);
            mat.convertTo(mat,CV_8UC3);
            cv::imwrite(path.string() + ".png",mat);
        }
        void utils::saveDepth(const Eigen::MatrixXd& img,const fs::path& path)
        {
            cv::Mat mat;
            cv::eigen2cv(img,mat);
            mat.convertTo(mat,CV_32F);
            cv::imwrite(path.string() + ".exr",mat);
        }

        void utils::writeTrajectory(const Trajectory& traj, const fs::path& path, bool writeCovariance)
        {
            std::fstream algoFile;
            algoFile.open(path,std::ios_base::out);
            algoFile << "# Algorithm Trajectory\n";
            algoFile << "# file: " << path << "\n";
            algoFile << "# timestamp tx ty tz qx qy qz qw\n";
            if(!algoFile.is_open()) { std::runtime_error("Could not open file at: " + path.string()); }

            for (const auto& pose : traj.poses())
            {
                Timestamp sec = (Timestamp)((double)pose.first/1e9);
                const auto t = pose.second->pose().translation();
                const auto q = pose.second->pose().unit_quaternion();
                algoFile << sec << "." << (Timestamp)((double)pose.first - (double)sec*1e9)  << " "
                << t.x() << " " << t.y() << " " << t.z() << " "
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w();
                if (writeCovariance)
                {
                    for (int i = 0; i < 6; i++)
                    {
                        for (int j = 0; j < 6; j++)
                        {
                            algoFile << " " << pose.second->cov()(i,j);
                        }
                    }
                }
                
                algoFile << "\n";
            }
            
        }


    }