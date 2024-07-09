#include "PAZ_Math"

double paz::gaussian_pdf(const Vec& mean, const Mat& cholCov, const Vec& x)
{
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

double paz::cs_divergence(const Vec& meanA, const Mat& cholCovA, const Vec&
    meanB, const Mat& cholCovB)
{
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
