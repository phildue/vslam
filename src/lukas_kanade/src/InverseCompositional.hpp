
#include "utils/utils.h"
#include "core/core.h"
#include <algorithm>
#include <execution>
namespace pd::vslam::lukas_kanade{

    template<typename Warp>
    InverseCompositional<Warp>::InverseCompositional (const Image& templ, const MatXd& dTx, const MatXd& dTy, const Image& image,
     std::shared_ptr<Warp> w0,
     least_squares::Loss::ShPtr l,
     double minGradient ,
     std::shared_ptr<const least_squares::Prior<Warp::nParameters>> prior)
    : least_squares::Problem<Warp::nParameters>()
    , _T(templ)
    , _I(image)
    , _w(w0)
    , _loss(l)
    , _prior(prior)
    {
        //TODO this could come from some external feature selector
        //TODO move dTx, dTy computation outside
        std::vector<Eigen::Vector2i> interestPoints;
        interestPoints.reserve(_T.rows() *_T.cols());
        for (int32_t v = 0; v < _T.rows(); v++)
        {
            for (int32_t u = 0; u < _T.cols(); u++)
            {
                if( std::sqrt(dTx(v,u)*dTx(v,u)+dTy(v,u)*dTy(v,u)) >= minGradient)
                {
                    interestPoints.emplace_back(u,v);
                }
                    
            }
        }
        _J.conservativeResize(_T.rows() *_T.cols(),Eigen::NoChange);
        _J.setZero();
        Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        size_t idx = 0U;
        std::for_each(interestPoints.begin(),interestPoints.end(),[&](auto kp)
            {
                const Eigen::Matrix<double, 2,nParameters> Jw = _w->J(kp.x(),kp.y());
                _J.row(kp.y() * _T.cols() + kp.x()) = Jw.row(0) * dTx(kp.y(), kp.x()) + Jw.row(1) * dTy(kp.y(),kp.x());
                const double Jnorm = _J.row(kp.y() * _T.cols() + kp.x()).norm();
                steepestDescent(kp.y(),kp.x()) = std::isfinite(Jnorm) ? Jnorm : 0.0;
                if (std::isfinite(Jnorm))
                {
                    _interestPoints.push_back({idx++,kp});
                }
            }
        );

        LOG_IMG("SteepestDescent") << steepestDescent;
    }
    
    template<typename Warp>
    InverseCompositional<Warp>::InverseCompositional (const Image& templ, const MatXd& dTx, const MatXd& dTy, const Image& image,
     std::shared_ptr<Warp> w0,
     const std::vector<Eigen::Vector2i>& interestPoints,
     least_squares::Loss::ShPtr l,
     std::shared_ptr<const least_squares::Prior<Warp::nParameters>> prior)
    : least_squares::Problem<Warp::nParameters>()
    , _T(templ)
    , _I(image)
    , _w(w0)
    , _loss(l)
    , _prior(prior)
    , _interestPoints(interestPoints.size())
    {
        _J.conservativeResize(_T.rows() *_T.cols(),Eigen::NoChange);
        _J.setZero();
        Eigen::MatrixXd steepestDescent = Eigen::MatrixXd::Zero(_T.rows(),_T.cols());
        std::atomic<size_t> idx = 0U;
        std::for_each(std::execution::par_unseq, interestPoints.begin(),interestPoints.end(),[&](auto kp)
            {
                const Eigen::Matrix<double, 2,nParameters> Jw = _w->J(kp.x(),kp.y());
                const Eigen::Matrix<double, 1,nParameters> Jwi = Jw.row(0) * dTx(kp.y(), kp.x()) + Jw.row(1) * dTy(kp.y(),kp.x());
                const double Jwin = Jwi.norm();
                if (std::isfinite(Jwin))
                {
                    _J.row(idx) = Jwi;
                    steepestDescent(kp.y(),kp.x()) = Jwin;
                    _interestPoints[idx] = {idx,kp};
                    idx++;
                }
            }
        );
        _J.conservativeResize(idx,Eigen::NoChange);
        _interestPoints.resize(idx);

        LOG_IMG("SteepestDescent") << steepestDescent;
    }
    template<typename Warp>
    InverseCompositional<Warp>::InverseCompositional (const Image& templ, const Image& image,std::shared_ptr<Warp> w0, std::shared_ptr<least_squares::Loss> l, double minGradient, std::shared_ptr<const least_squares::Prior<Warp::nParameters>> prior)
    : InverseCompositional<Warp> (templ, algorithm::gradX(templ).cast<double>(), algorithm::gradY(templ).cast<double>(), image, w0, l, minGradient,prior){}



    template<typename Warp>
    void InverseCompositional<Warp>::updateX(const Eigen::Matrix<double,Warp::nParameters,1>& dx)
    {
        _w->updateCompositional(-dx);
    }
    template<typename Warp>
    typename least_squares::NormalEquations<Warp::nParameters>::ConstShPtr InverseCompositional<Warp>::computeNormalEquations() 
    {
        Image IWxp = Image::Zero(_I.rows(),_I.cols());
        MatXd R = MatXd::Zero(_I.rows(),_I.cols());
        MatXd W = MatXd::Zero(_I.rows(),_I.cols());
        VecXd r = VecXd::Zero(_interestPoints.size());
        VecXd w = VecXd::Zero(_interestPoints.size());

        std::for_each(std::execution::par_unseq,_interestPoints.begin(),_interestPoints.end(),
        [&](auto kp) {
                Eigen::Vector2d uvI = _w->apply(kp.pos.x(),kp.pos.y());
                const bool visible = 1 < uvI.x() && uvI.x() < _I.cols() - 1 && 1 < uvI.y() && uvI.y() < _I.rows() - 1 && std::isfinite(uvI.x());
                if (visible){
                    //IWxp(kp.pos.y(),kp.pos.x()) = algorithm::bilinearInterpolation(_I,uvI.x(),uvI.y());
                    IWxp(kp.pos.y(),kp.pos.x()) = _I((int)  std::round(uvI.y()),(int) std::round(uvI.x()));
                    R(kp.pos.y(),kp.pos.x()) = (double)IWxp(kp.pos.y(),kp.pos.x()) - (double)_T(kp.pos.y(),kp.pos.x());
                    W(kp.pos.y(),kp.pos.x()) = 1.0;
                    r(kp.idx) = R(kp.pos.y(),kp.pos.x());
                    w(kp.idx) = W(kp.pos.y(),kp.pos.x());
                }
            }
        );
        
        if(_loss)
        {  
            _loss->computeScale(r);
            std::for_each(std::execution::par_unseq,_interestPoints.begin(),_interestPoints.end(),
            [&](auto kp) 
                {
                    if(w(kp.idx) > 0.0){
                        W(kp.pos.y(),kp.pos.x()) = _loss->computeWeight( R(kp.pos.y(),kp.pos.x()));
                        w(kp.idx) = W(kp.pos.y(),kp.pos.x());
                    }
                    
                }
            );
            
        }
        auto ne = std::make_shared<least_squares::NormalEquations<nParameters>>();
        auto Jtw = _J.transpose() * w.asDiagonal();
        ne->A = Jtw * _J;
        ne->b = Jtw * r;
        ne->chi2 = (r * w).transpose() * r;
        ne->nConstraints = r.rows();
        ne->A.noalias() = ne->A / (double)ne->nConstraints;
        ne->b.noalias() = ne->b / (double)ne->nConstraints;

        if (_prior){ _prior->apply(ne,_w->x()); }

        LOG_IMG("ImageWarped") << IWxp;
        LOG_IMG("Residual") << R;
        LOG_IMG("Weights") << W;

        return ne;


    }

}
