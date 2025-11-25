#include "PAZ_Math"
#include <cstdint>

static void fft_internal(paz::complex* samples, std::size_t numSamples, bool
    invert)
{
    if(numSamples < 2)
    {
        return;
    }

    std::vector<paz::complex> evens(numSamples/2);
    for(std::size_t i = 0; i < numSamples/2; ++i)
    {
        evens[i] = samples[2*i];
    }
    fft_internal(evens.data(), evens.size(), invert);

    std::vector<paz::complex> odds(numSamples/2);
    for(std::size_t i = 0; i < numSamples/2; ++i)
    {
        odds[i] = samples[2*i + 1];
    }
    fft_internal(odds.data(), odds.size(), invert);

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

paz::ComplexVec paz::fft(const ComplexMatRef& timeSamples)
{
    if(timeSamples.rows() != 1 && timeSamples.cols() != 1)
    {
        throw std::runtime_error("Not a vector.");
    }

    std::size_t numSamples = timeSamples.size();

    if(numSamples < 2)
    {
        return timeSamples;
    }

    ComplexVec freqSamples = timeSamples;

    const std::size_t paddedSize = next_pow2(numSamples);
    if(numSamples != paddedSize)
    {
        throw std::logic_error("Non-power-of-two not implemented.");
    }

    fft_internal(freqSamples.data(), numSamples, false);

    std::reverse(freqSamples.begin() + 1, freqSamples.end());

    return freqSamples;
}

paz::ComplexVec paz::ifft(const ComplexMatRef& freqSamples)
{
    if(freqSamples.rows() != 1 && freqSamples.cols() != 1)
    {
        throw std::runtime_error("Not a vector.");
    }

    std::size_t numSamples = freqSamples.size();

    if(numSamples < 2)
    {
        return freqSamples;
    }

    ComplexVec timeSamples = freqSamples;

    const std::size_t paddedSize = next_pow2(numSamples);
    if(numSamples != paddedSize)
    {
        throw std::logic_error("Non-power-of-two not implemented.");
    }

    std::reverse(timeSamples.begin() + 1, timeSamples.end());

    fft_internal(timeSamples.data(), numSamples, true);

    return timeSamples;
}

paz::ComplexVec paz::fftshift(const ComplexMatRef& freqSamples)
{
    if(freqSamples.rows() != 1 || freqSamples.cols() != 1)
    {
        throw std::runtime_error("Not a vector.");
    }

    const std::size_t numSamples = freqSamples.size();
    ComplexVec res(numSamples);
    for(std::size_t i = 0; i < numSamples; ++i)
    {
        res(i) = freqSamples((i + (numSamples + 1)/2)%numSamples);
    }
    return res;
}

paz::ComplexVec paz::ifftshift(const ComplexMatRef& freqSamples)
{
    if(freqSamples.rows() != 1 || freqSamples.cols() != 1)
    {
        throw std::runtime_error("Not a vector.");
    }

    const std::size_t numSamples = freqSamples.size();
    ComplexVec res(numSamples);
    for(std::size_t i = 0; i < numSamples; ++i)
    {
        res(i) = freqSamples((i + numSamples/2)%numSamples);
    }
    return res;
}
