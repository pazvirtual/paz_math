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

paz::Vec::Vec(std::size_t rows) : Mat(rows, 1) {}

paz::Vec::Vec(const Mat& m) : Mat(m)
{
    if(rows() && cols() != 1)
    {
        throw std::runtime_error("Matrix must be a column vector or empty.");
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

paz::Vec paz::Vec::segment(std::size_t start, std::size_t n) const
{
    return block(start, 0, n, 1);
}

paz::Vec paz::Vec::head(std::size_t n) const
{
    return segment(0, n);
}

paz::Vec paz::Vec::tail(std::size_t n) const
{
    return segment(size() - 1 - n, n);
}

paz::Vec paz::Vec::cross(const Vec& rhs) const
{
    if(size() != 3 || rhs.size() != 3)
    {
        throw std::runtime_error("Not a 3-vector.");
    }
    return {{(*this)(1)*rhs(2) - (*this)(2)*rhs(1),
             (*this)(2)*rhs(0) - (*this)(0)*rhs(2),
             (*this)(0)*rhs(1) - (*this)(1)*rhs(0)}};
}
