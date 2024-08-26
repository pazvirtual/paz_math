#include "PAZ_Math"

double paz::gaussian_pdf(const MatRef& mean, const MatRef& cholCov, const
    MatRef& x)
{
    if(mean.cols() != 1 || x.cols() != 1)
    {
        throw std::runtime_error("Not a column vector.");
    }
    double p = 1.;
    for(std::size_t i = 0; i < mean.size(); ++i)
    {
        p *= TwoPi;
    }
    const double detCholCov = cholCov.det();
    const double detCov = detCholCov*detCholCov;
    if(!detCov)
    {
        throw std::runtime_error("Covariance is not invertible.");
    }
    return std::exp(-0.5*(cholCov.inv()*(x - mean)).normSq())/std::sqrt(p*
        detCov);
}

double paz::cs_divergence(const MatRef& meanA, const MatRef& cholCovA, const
    MatRef& meanB, const MatRef& cholCovB)
{
    if(meanA.cols() != 1 || meanB.cols() != 1)
    {
        throw std::runtime_error("Not a column vector.");
    }
    double p2 = 1.;
    for(std::size_t i = 0; i < meanA.size(); ++i)
    {
        p2 *= 2.*TwoPi;
    }
    const Mat covA = cholCovA*cholCovA.trans();
    const Mat covB = cholCovB*cholCovB.trans();
    const Mat cholCov = (covA + covB).chol();
    return -std::log(gaussian_pdf(meanB, cholCov, meanA)) - 0.25*std::log(p2*
        covA.det()) - 0.25*std::log(p2*covB.det());
}

paz::Vec paz::gmm_rand(const std::vector<double>& weights, const std::vector<
    Vec>& means, const std::vector<Mat>& cholCovs)
{
    const std::size_t idx = pmf_rand(weights);
    return cholCovs[idx]*Vec::Randn(means[idx].size()) + means[idx];
}
