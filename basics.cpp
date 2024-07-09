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

int paz::randi(int a, int b)
{
    std::uniform_int_distribution<int> dis(a, b);
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
