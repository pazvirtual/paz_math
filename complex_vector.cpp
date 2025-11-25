#include "PAZ_Math"

paz::ComplexVec paz::ComplexVec::Constant(std::size_t rows, complex c)
{
    ComplexVec v(rows);
    std::fill(v.begin(), v.end(), c);
    return v;
}

paz::ComplexVec paz::ComplexVec::Zero(std::size_t rows)
{
    return Constant(rows, 0.);
}

paz::ComplexVec paz::ComplexVec::Ones(std::size_t rows)
{
    return Constant(rows, 1.);
}

paz::ComplexVec paz::ComplexVec::IdQuat()
{
    return {{0., 0., 0., 1.}};
}

paz::ComplexVec paz::ComplexVec::Randn(std::size_t rows)
{
    ComplexVec v(rows);
    for(auto& n : v)
    {
        n = randn();
    }
    return v;
}

paz::ComplexVec paz::ComplexVec::Cat(const ComplexMatRef& a, const
    ComplexMatRef& b)
{
    if(a.cols() != 1 || b.cols() != 1)
    {
        throw std::runtime_error("Matrices must be column vectors.");
    }
    ComplexVec v(a.size() + b.size());
    std::copy(a.begin(), a.end(), v.begin());
    std::copy(b.begin(), b.end(), v.begin() + a.size());
    return v;
}

paz::ComplexVec::ComplexVec(std::size_t rows) : ComplexMat(rows, 1) {}

paz::ComplexVec::ComplexVec(const ComplexMat& m) : ComplexMat(m)
{
    if(cols() != 1)
    {
        throw std::runtime_error("Matrix must be a column vector.");
    }
}

paz::ComplexVec::ComplexVec(const ComplexMatRef& m) : ComplexMat(m)
{
    if(cols() != 1)
    {
        throw std::runtime_error("Matrix must be a column vector.");
    }
}

paz::ComplexVec::ComplexVec(const std::initializer_list<std::initializer_list<
    complex>>& list)
{
    if(!list.size())
    {
        return;
    }
    if(list.size() != 1)
    {
        throw std::runtime_error("Vector initializer list is malformed.");
    }
    *this = ComplexVec(list.begin()->size());
    std::copy(list.begin()->begin(), list.begin()->end(), begin());
}

paz::ComplexMatRef paz::ComplexVec::segment(std::size_t start, std::size_t n)
    const
{
    return block(start, 0, n, 1);
}

void paz::ComplexVec::setSegment(std::size_t start, std::size_t n, const
    ComplexMatRef& rhs)
{
    setBlock(start, 0, n, 1, rhs);
}

paz::ComplexMatRef paz::ComplexVec::head(std::size_t n) const
{
    return segment(0, n);
}

void paz::ComplexVec::setHead(std::size_t n, const ComplexVec& rhs)
{
    setSegment(0, n, rhs);
}

paz::ComplexMatRef paz::ComplexVec::tail(std::size_t n) const
{
    return segment(size() - n, n);
}

void paz::ComplexVec::setTail(std::size_t n, const ComplexVec& rhs)
{
    setSegment(size() - n, n, rhs);
}

void paz::ComplexVec::resize(std::size_t newRows)
{
    resizeRows(newRows);
}
