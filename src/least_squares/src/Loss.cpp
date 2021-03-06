#include "Loss.h"
#include "core/core.h"
#include "utils/utils.h"

namespace pd::vslam::least_squares{


    double TukeyLoss::compute(double r) const
    {
        const double rs = scale(r);
        // If the residuals falls within the 95% the loss is quadratic
        if ( std::abs(rs) < TukeyLoss::C)
        {
        const double r_c = rs/TukeyLoss::C;

        return C2_6*(1.0-std::pow( 1.0 - r_c*r_c,3));

        }else{
        // Outliers are disregarded
        return C2_6;
        }
    }

    double TukeyLoss::computeDerivative(double r) const
    {
        const double rs = scale(r);

        // If the residuals falls within the 95% the loss is quadratic
        if ( std::abs(rs) < TukeyLoss::C)
        {
                const double r_c = rs/TukeyLoss::C;

                return rs*std::pow( 1.0 - r_c*r_c,2);

        }else{
                // Outliers are disregarded
                return 0.0;
        }
    }

    double TukeyLoss::computeWeight(double r) const
    {
        const double rs = scale(r);

        // If the residuals falls within the 95% the loss is quadratic
        if ( std::abs(rs) < TukeyLoss::C)
        {
                const double r_c = rs/TukeyLoss::C;

                return std::pow( 1.0 - r_c*r_c,2);

        }else{
                // Outliers are disregarded
                return 0.0;
        }
    }

    double HuberLoss::computeWeight(double r) const
        {
                const double rs = scale(r);

                if(std::abs(rs) < _c )
                {
                        return 1.0;
                }else{
                        return (_c * rs > 0.0 ? 1.0 : -1.0)/rs;
                }
        }
        //dl/dr
        double HuberLoss::computeDerivative(double r) const
        {
                const double rs = scale(r);
                if(std::abs(rs) < _c )
                {
                        return rs;
                }else{
                        return _c * rs > 0.0 ? 1.0 : -1.0;
                }
        }
        //l(r)
        double HuberLoss::compute(double r) const
        {
                const double rs = scale(r);
                if(std::abs(rs) < _c )
                {
                        return 0.5 * rs*rs;
                }else{
                        return _c * std::abs(rs) - 0.5 * rs*rs;
                }
        }

        double LossTDistribution::computeWeight(double r) const
        {
                const double rs = scale(r);

                return (_v + 1.0) / (_v + rs*rs);
        }
        double LossTDistribution::computeDerivative(double UNUSED(r)) const
        {
                return 0.0; //TODO
        }
        double LossTDistribution::compute(double UNUSED(r)) const
        {
                return 0.0; //TODO

        }
   
     
}