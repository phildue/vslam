//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include <core/core.h>
#include <utils/utils.h>
#include <opencv2/highgui.hpp>
#include <lukas_kanade/lukas_kanade.h>
using namespace testing;
using namespace pd;
using namespace pd::vslam;
using namespace pd::vslam::lukas_kanade;

#define VISUALIZE true
TEST(FrameTest,CreatePyramid)
{
    DepthMap depth = utils::loadDepth(TEST_RESOURCE"/depth.png")/5000.0;
    Image img = utils::loadImage(TEST_RESOURCE"/rgb.png");
   
    auto cam = std::make_shared<Camera>(525.0,525.0,319.5,239.5);
    auto f = std::make_shared<FrameRgbd>(img,depth,cam,3,0);
    for(size_t i = 0; i < f->nLevels(); i++)
    {
        auto pcl = f->pcl(i,true);
        DepthMap depthReproj = algorithm::resize(depth,std::pow(0.5,i));
        
        for(const auto& p : pcl)
        {
            const Eigen::Vector2i uv = f->camera2image(p,i).cast<int>();
            EXPECT_GT(uv.x(),0);
            EXPECT_GT(uv.y(),0);
            EXPECT_LT(uv.x(),f->width(i));
            EXPECT_LT(uv.y(),f->height(i));

            depthReproj(uv.y(),uv.x()) = p.z();
        }

        EXPECT_NEAR((depthReproj - f->depth(i)).norm(), 0.0, 1e-6);

        depthReproj = algorithm::resize(depth,std::pow(0.5,i));
        
        pcl = f->pcl(i,false);
        for(const auto& p : pcl)
        {
            const Eigen::Vector2i uv = f->camera2image(p,i).cast<int>();
            if(0 <= uv.x() && uv.x() < depthReproj.cols() &&
               0 <= uv.y() && uv.y() < depthReproj.cols()
            )
            {
                depthReproj(uv.y(),uv.x()) = p.z();
            }

        }

        EXPECT_NEAR((depthReproj - f->depth(i)).norm(), 0.0, 1e-6);
        
        if (VISUALIZE)
        {
            cv::imshow("Image",vis::drawMat(f->intensity(i)));
            cv::imshow("dIx",vis::drawAsImage(f->dIx(i).cast<double>()));
            cv::imshow("dIy",vis::drawAsImage(f->dIy(i).cast<double>()));
            cv::imshow("Depth Reproj",vis::drawAsImage(depthReproj));
            cv::imshow("Depth",vis::drawAsImage(f->depth(i)));
            cv::waitKey(0);
        }
        
    }
}

TEST(WarpTest,Warp)
{
    DepthMap depth = utils::loadDepth(TEST_RESOURCE"/depth.png")/5000.0;
    Image img = utils::loadImage(TEST_RESOURCE"/rgb.png");
   
    auto cam = std::make_shared<Camera>(525.0,525.0,319.5,239.5);
    auto f0 = std::make_shared<FrameRgbd>(img,depth,cam,3,0);
    auto f1 = std::make_shared<FrameRgbd>(img,depth,cam,3,0);
 
    for(size_t i = 0; i < f0->nLevels(); i++)
    {
         auto w = std::make_shared<WarpSE3>(f0->pose().pose(),f0->pcl(i,false),f0->width(i),
                        f0->camera(i),f1->camera(i),f1->pose().pose());
        
        if (VISUALIZE)
        {
            auto& img = f0->intensity(i);
            Image iwxp = w->apply(img);
            Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(iwxp.rows(),iwxp.cols());
            std::vector<MatXd> Js(6,MatXd::Zero(iwxp.rows(),iwxp.cols()));
            for(int v = 0; v < steepestDescent.rows(); v++)
            {
                for(int u = 0; u < steepestDescent.cols(); u++)
                {
                    const Eigen::Matrix<double, 2,6> Jw = w->J(u,v);
                    const Eigen::Matrix<double, 1,6> Jwi = Jw.row(0) * f0->dIx(i)(v, u) + Jw.row(1) * f0->dIx(i)(v, u);
                    //std::cout << "J = " << Jwi << std::endl;
                    for(int j = 0; j < 6; j++){ Js[j](v,u) = Jwi(j); }

                    steepestDescent(v,u) = std::isfinite(Jwi.norm()) ? Jwi.norm() : 0.0;

                }
            }
            for(int j = 0; j < 6 ;j++)
            {
                cv::imshow("J"+std::to_string(j),vis::drawAsImage(Js.at(j)));
            }
            cv::imshow("Image",vis::drawMat(f0->intensity(i)));
            cv::imshow("Iwxp",vis::drawMat(iwxp));
            cv::imshow("J",vis::drawAsImage(steepestDescent));
            cv::waitKey(0);
        }
        
    }
}
