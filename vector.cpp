#include "PAZ_Math"

paz::Vec paz::Vec::Constant(std::size_t rows, double c)
{
    Vec v(rows);
    std::fill(v.begin(), v.end(), c);
    return v;
}

paz::Vec paz::Vec::Zero(std::size_t rows)
{
    return Constant(rows, 0.);
}

paz::Vec paz::Vec::Ones(std::size_t rows)
{
    return Constant(rows, 1.);
}

paz::Vec paz::Vec::IdQuat()
{
    return {{0., 0., 0., 1.}};
}

paz::Vec paz::Vec::Randn(std::size_t rows)
{
    Vec v(rows);
    for(auto& n : v)
    {
        n = randn();
    }
    return v;
}

paz::Vec paz::Vec::Cat(const MatRef& a, const MatRef& b)
{
    if(a.cols() != 1 || b.cols() != 1)
    {
        throw std::runtime_error("Matrices must be column vectors.");
    }
    Vec v(a.size() + b.size());
    std::copy(a.begin(), a.end(), v.begin());
    std::copy(b.begin(), b.end(), v.begin() + a.size());
    return v;
}

paz::Vec::Vec(std::size_t rows) : Mat(rows, 1) {}

paz::Vec::Vec(const Mat& m) : Mat(m)
{
    if(cols() != 1)
    {
        throw std::runtime_error("Matrix must be a column vector.");
    }
}

paz::Vec::Vec(const MatRef& m) : Mat(m)
{
    if(cols() != 1)
    {
        throw std::runtime_error("Matrix must be a column vector.");
    }
}

paz::Vec::Vec(const std::initializer_list<std::initializer_list<double>>& list)
{
    if(!list.size())
    {
        return;
    }
    if(list.size() != 1)
    {
        throw std::runtime_error("Vector initializer list is malformed.");
    }
    *this = Vec(list.begin()->size());
    std::copy(list.begin()->begin(), list.begin()->end(), begin());
}

paz::MatRef paz::Vec::segment(std::size_t start, std::size_t n) const
{
    return block(start, 0, n, 1);
}

void paz::Vec::setSegment(std::size_t start, std::size_t n, const MatRef& rhs)
{
    setBlock(start, 0, n, 1, rhs);
}

paz::MatRef paz::Vec::head(std::size_t n) const
{
    return segment(0, n);
}

void paz::Vec::setHead(std::size_t n, const Vec& rhs)
{
    setSegment(0, n, rhs);
}

paz::MatRef paz::Vec::tail(std::size_t n) const
{
    return segment(size() - n, n);
}

void paz::Vec::setTail(std::size_t n, const Vec& rhs)
{
    setSegment(size() - n, n, rhs);
}

void paz::Vec::resize(std::size_t newRows)
{
    resizeRows(newRows);
}
