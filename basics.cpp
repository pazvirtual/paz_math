#include "PAZ_Math"
#include <random>
#include <thread>

static std::mt19937_64 RandomEngine(std::chrono::system_clock::now().
    time_since_epoch().count()^std::hash<std::thread::id>{}(std::this_thread::
    get_id()));

double paz::randn()
{
    std::normal_distribution<double> dis(0., 1.);
    return dis(RandomEngine);
}

double paz::uniform()
{
    std::uniform_real_distribution<double> dis(0., 1.);
    return dis(RandomEngine);
}

double paz::uniform(double a, double b)
{
    std::uniform_real_distribution<double> dis(a, b);
    return dis(RandomEngine);
}
