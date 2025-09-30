#include "PAZ_Math"
#include <random>
#include <thread>
#include <sstream>

static std::mt19937_64& random_engine()
{
    thread_local auto engine = std::mt19937_64{paz::random_seed()};
    return engine;
}

std::size_t paz::random_seed()
{
    thread_local const std::size_t seed = std::chrono::system_clock::now().
        time_since_epoch().count()^std::hash<std::thread::id>{}(std::this_thread
        ::get_id());
    return seed;
}

double paz::randn()
{
    std::normal_distribution<double> dis(0., 1.);
    return dis(random_engine());
}

std::size_t paz::randi(std::size_t n)
{
    if(!n)
    {
        return 0;
    }
    std::uniform_int_distribution<std::size_t> dis(0, n);
    return dis(random_engine());
}

double paz::uniform()
{
    std::uniform_real_distribution<double> dis(0., 1.);
    return dis(random_engine());
}

double paz::uniform(double a, double b)
{
    std::uniform_real_distribution<double> dis(a, b);
    return dis(random_engine());
}

std::size_t paz::pmf_rand(const std::vector<double>& probs)
{
    const std::size_t n = probs.size();
    if(!n)
    {
        throw std::runtime_error("PMF has no support.");
    }
    if(n == 1)
    {
        return 0;
    }
    const double u = uniform();
    double sum = 0.;
    for(std::size_t i = 0; i < probs.size(); ++i)
    {
        sum += probs[i];
        if(sum > u)
        {
            return i;
        }
    }
    std::ostringstream oss;
    oss << "Sum of PMF probabilities (" << sum << ") is less than one.";
    throw std::runtime_error(oss.str());
}

std::vector<std::size_t> paz::rand_seq(std::size_t length)
{
    std::vector<std::size_t> indices(length);
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    std::shuffle(indices.begin(), indices.end(), random_engine());
    return indices;
}

std::size_t paz::poissrnd(double lambda)
{
    std::poisson_distribution<std::size_t> dis(lambda);
    return dis(random_engine());
}

void paz::normalize_log_weights(std::vector<double>& logWeights)
{
    const double maxLogWeight = *std::max_element(logWeights.begin(),
        logWeights.end());
    double logSum = 0.;
    for(auto n : logWeights)
    {
        logSum += std::exp(n - maxLogWeight);
    }
    logSum = std::log(logSum) + maxLogWeight;
    for(auto& n : logWeights)
    {
        n = std::exp(n - logSum);
    }
}

void paz::normalize_weights(std::vector<double>& weights)
{
    for(auto& n : weights)
    {
        n = std::log(n);
    }
    normalize_log_weights(weights);
}

paz::Mat paz::rot1(double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return {{1., 0., 0.},
            {0.,  c,  s},
            {0., -s,  c}};
}

paz::Mat paz::rot2(double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return {{ c, 0., -s},
            {0., 1., 0.},
            { s, 0.,  c}};
}

paz::Mat paz::rot3(double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return {{ c,  s, 0.},
            {-s,  c, 0.},
            {0., 0., 1.}};
}

std::vector<double> paz::real(const std::vector<complex>& v)
{
    const std::size_t n = v.size();
    std::vector<double> res(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        res[i] = v[i].real();
    }
    return res;
}

std::vector<double> paz::imag(const std::vector<complex>& v)
{
    const std::size_t n = v.size();
    std::vector<double> res(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        res[i] = v[i].imag();
    }
    return res;
}

std::vector<double> paz::abs(const std::vector<complex>& v)
{
    const std::size_t n = v.size();
    std::vector<double> res(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        res[i] = std::abs(v[i]);
    }
    return res;
}

std::vector<double> paz::arg(const std::vector<complex>& v)
{
    const std::size_t n = v.size();
    std::vector<double> res(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        res[i] = std::arg(v[i]);
    }
    return res;
}

std::vector<paz::complex> paz::conj(const std::vector<complex>& v)
{
    const std::size_t n = v.size();
    std::vector<complex> res(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        res[i] = std::conj(v[i]);
    }
    return res;
}
