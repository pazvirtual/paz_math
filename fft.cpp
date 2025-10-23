#include "PAZ_Math"
#include <cstdint>

static void fft_internal(std::vector<paz::complex>& samples, bool invert)
{
    const std::size_t numSamples = samples.size();

    if(numSamples < 2)
    {
        return;
    }

    std::vector<paz::complex> evens(numSamples/2);
    for(std::size_t i = 0; i < numSamples/2; ++i)
    {
        evens[i] = samples[2*i];
    }
    fft_internal(evens, invert);

    std::vector<paz::complex> odds(numSamples/2);
    for(std::size_t i = 0; i < numSamples/2; ++i)
    {
        odds[i] = samples[2*i + 1];
    }
    fft_internal(odds, invert);

    const double angle = (invert ? -1. : 1.)*paz::TwoPi/numSamples;
    const paz::complex t0 = {std::cos(angle), std::sin(angle)};
    paz::complex t1 = 1.;
    for(std::size_t i = 0; i < numSamples/2; ++i)
    {
        samples[i] = evens[i] + t1*odds[i];
        samples[i + numSamples/2] = evens[i] - t1*odds[i];
        if(invert)
        {
            samples[i] *= 0.5;
            samples[i + numSamples/2] *= 0.5;
        }
        t1 *= t0;
    }
}

static std::uint64_t next_pow2(std::uint64_t x)
{
    if(!x)
    {
        return 1;
    }
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    ++x;
    return x;
}

std::vector<paz::complex> paz::fft(const std::vector<complex>& timeSamples)
{
    std::size_t numSamples = timeSamples.size();

    if(numSamples < 2)
    {
        return timeSamples;
    }

    std::vector<complex> freqSamples = timeSamples;

    const std::size_t paddedSize = next_pow2(numSamples);
    if(numSamples != paddedSize)
    {
        throw std::logic_error("Non-power-of-two not implemented.");
    }

    fft_internal(freqSamples, false);

    std::reverse(freqSamples.begin() + 1, freqSamples.end());

    return freqSamples;
}

std::vector<paz::complex> paz::ifft(const std::vector<complex>& freqSamples)
{
    std::size_t numSamples = freqSamples.size();

    if(numSamples < 2)
    {
        return freqSamples;
    }

    std::vector<complex> timeSamples = freqSamples;

    const std::size_t paddedSize = next_pow2(numSamples);
    if(numSamples != paddedSize)
    {
        throw std::logic_error("Non-power-of-two not implemented.");
    }

    std::reverse(timeSamples.begin() + 1, timeSamples.end());

    fft_internal(timeSamples, true);

    return timeSamples;
}

std::vector<paz::complex> paz::fftshift(const std::vector<complex>& freqSamples)
{
    const std::size_t numSamples = freqSamples.size();
    std::vector<complex> res(numSamples);
    for(std::size_t i = 0; i < numSamples; ++i)
    {
        res[i] = freqSamples[(i + (numSamples + 1)/2)%numSamples];
    }
    return res;
}

std::vector<paz::complex> paz::ifftshift(const std::vector<complex>&
    freqSamples)
{
    const std::size_t numSamples = freqSamples.size();
    std::vector<complex> res(numSamples);
    for(std::size_t i = 0; i < numSamples; ++i)
    {
        res[i] = freqSamples[(i + numSamples/2)%numSamples];
    }
    return res;
}
