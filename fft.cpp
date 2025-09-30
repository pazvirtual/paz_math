#include "PAZ_Math"

//TEMP - replace (I)DFT with (I)FFT

std::vector<paz::complex> paz::fft(const std::vector<complex>& timeSamples)
{
    const std::size_t numSamples = timeSamples.size();
    std::vector<complex> freqSamples(numSamples, 0.);

    for(std::size_t i = 0; i < numSamples; ++i)
    {
        for(std::size_t j = 0; j < numSamples; ++j)
        {
            freqSamples[i] += timeSamples[j]*std::exp(-ImagUnit*TwoPi*
                static_cast<double>(i)*static_cast<double>(j)*(1./numSamples));
        }
    }

    return freqSamples;
}

std::vector<paz::complex> paz::ifft(const std::vector<complex>& freqSamples)
{
    const std::size_t numSamples = freqSamples.size();
    std::vector<complex> timeSamples(numSamples, 0.);

    for(std::size_t i = 0; i < numSamples; ++i)
    {
        for(std::size_t j = 0; j < numSamples; ++j)
        {
            timeSamples[i] += freqSamples[j]*std::exp(ImagUnit*TwoPi*
                static_cast<double>(i)*static_cast<double>(j)*(1./numSamples));
        }
        timeSamples[i] /= numSamples;
    }

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
