//
// Created by phil on 10.10.20.
//

#include <gtest/gtest.h>
#include "utils/utils.h"
#include "core/core.h"
#include "least_squares/least_squares.h"
#include "lukas_kanade/lukas_kanade.h"

using namespace testing;
using namespace pd;
using namespace pd::vslam;
using namespace pd::vslam::least_squares;
using namespace pd::vslam::lukas_kanade;

#define VISUALIZE true

class LukasKanadeOpticalFlowTest : public Test{
    public:
    Image img0,img1;
    Eigen::Matrix3d A;
    int _nRuns = 20;
    int _nFailed = 0;
    LukasKanadeOpticalFlowTest()
    {
        img0 = utils::loadImage(TEST_RESOURCE"/person.jpg",50,50,true);
        A = Eigen::Matrix3d::Identity();
        img1 = img0;
        algorithm::warpAffine(img0,A,img1);
    
    }
};

TEST_F(LukasKanadeOpticalFlowTest,LukasKanadeOpticalFlow)
{

    for (int i = 0; i < _nRuns; i++)
    {
        Eigen::Vector2d x;
        x << random::U(5,6)*random::sign(),random::U(5,6)*random::sign();
        auto w = std::make_shared<WarpOpticalFlow>(x);
        auto gn = std::make_shared<GaussNewton<InverseCompositionalOpticalFlow::nParameters>> ( 1e-7,100);
        auto lk = std::make_shared<InverseCompositionalOpticalFlow> (img1,img0,w);
        if (VISUALIZE)
        {
            LOG_IMG("ImageWarped")->_show = true;
            LOG_IMG("Depth")->_show = true;
            LOG_IMG("Residual")->_show = true;
            LOG_IMG("Image")->_show = true;
            LOG_IMG("Depth")->_show = true;
            LOG_IMG("Weights")->_show = true;
        }
        
        ASSERT_GT(w->x().norm(), 1.0) << "Noise should be greater than that.";

        gn->solve(lk);
        
        if (w->x().norm() > 1.0){_nFailed++;}
    }

    EXPECT_LE((double)_nFailed/(double)_nRuns,0.05) << "Majority of test cases should pass.";
     
}


