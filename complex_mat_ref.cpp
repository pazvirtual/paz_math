#include "PAZ_Math"


paz::ComplexMatRef::ComplexMatRef(const complex* ptr, std::size_t origRows,
    std::size_t origCols, std::size_t blockRows, std::size_t blockCols) :
    _begin({ptr, 0, static_cast<iterator::difference_type>(origRows),
    static_cast<iterator::difference_type>(blockRows)}), _origCols(origCols),
    _blockCols(blockCols) {}

paz::ComplexMatRef::ComplexMatRef(const ComplexMat& m) : ComplexMatRef(m.data(),
    m.rows(), m.cols(), m.rows(), m.cols()) {}

paz::complex paz::ComplexMatRef::det() const
{
    return ComplexMat(*this).det();
}

paz::ComplexMat paz::ComplexMatRef::inv() const
{
    return ComplexMat(*this).inv();
}

paz::ComplexMat paz::ComplexMatRef::solve(const ComplexMat& b) const
{
    return ComplexMat(*this).solve(b);
}

paz::ComplexMat paz::ComplexMatRef::chol() const
{
    return ComplexMat(*this).chol();
}

paz::ComplexMat paz::ComplexMatRef::cholUpdate(const ComplexMat& m, double a)
    const
{
    return ComplexMat(*this).cholUpdate(m, a);
}

paz::ComplexVec paz::ComplexMatRef::eig() const
{
    return ComplexMat(*this).eig();
}

paz::ComplexVec paz::ComplexMatRef::eig(ComplexMat& vecs) const
{
    return ComplexMat(*this).eig(vecs);
}

void paz::ComplexMatRef::qr(ComplexMat& q, ComplexMat& r) const
{
    ComplexMat(*this).qr(q, r);
}

void paz::ComplexMatRef::qr(ComplexMat& q, ComplexMat& r, std::vector<std::
    size_t>& p) const
{
    ComplexMat(*this).qr(q, r, p);
}

paz::ComplexMat paz::ComplexMatRef::trans() const
{
    return ComplexMat(*this).trans();
}

paz::ComplexVec paz::ComplexMatRef::diag() const
{
    if(rows() != cols() || empty())
    {
        throw std::runtime_error("Matrix must be square.");
    }
    ComplexVec res(rows());
    for(std::size_t i = 0; i < rows(); ++i)
    {
        res(i) = operator()(i, i);
    }
    return res;
}

paz::ComplexMat paz::ComplexMatRef::rep(std::size_t m, std::size_t n) const
{
    ComplexMat res(m*rows(), n*cols());
    for(std::size_t i = 0; i < m*n; ++i)
    {
        std::copy(begin(), end(), res.begin() + rows()*cols()*i);
    }
    return res;
}

paz::complex paz::ComplexMatRef::normSq() const
{
    return dot(*this);
}

paz::complex paz::ComplexMatRef::norm() const
{
    return std::sqrt(normSq());
}

paz::complex paz::ComplexMatRef::sum() const
{
    return std::accumulate(begin(), end(), complex{0.});
}

paz::ComplexVec paz::ComplexMatRef::rowSum() const
{
    ComplexVec res = ComplexVec::Zero(rows());
    for(std::size_t i = 0; i < rows(); ++i)
    {
        for(std::size_t j = 0; j < cols(); ++j)
        {
            res(i) += operator()(i, j);
        }
    }
    return res;
}

paz::ComplexMat paz::ComplexMatRef::colSum() const
{
    ComplexMat res = ComplexMat::Zero(1, cols());
    for(std::size_t i = 0; i < cols(); ++i)
    {
        for(std::size_t j = 0; j < rows(); ++j)
        {
            res(0, i) += operator()(j, i);
        }
    }
    return res;
}

paz::complex paz::ComplexMatRef::min() const
{
    return *std::min_element(begin(), end(), [](complex a, complex b){ return
        std::abs(a) < std::abs(b); });
}

paz::complex paz::ComplexMatRef::max() const
{
    return *std::max_element(begin(), end(), [](complex a, complex b){ return
        std::abs(a) > std::abs(b); });
}

paz::ComplexMat paz::ComplexMatRef::normalized() const
{
    ComplexMat m = *this;
    m /= norm();
    return m;
}

paz::ComplexMat paz::ComplexMatRef::prod(const ComplexMatRef& rhs) const // elementwise
{
    if(rows() != rhs.rows() || cols() != rhs.cols())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    ComplexMat m = *this;
    for(std::size_t i = 0; i < size(); ++i)
    {
        m(i) *= rhs(i);
    }
    return m;
}

paz::ComplexMat paz::ComplexMatRef::quot(const ComplexMatRef& rhs) const // elementwise
{
    if(rows() != rhs.rows() || cols() != rhs.cols())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    ComplexMat m = *this;
    for(std::size_t i = 0; i < size(); ++i)
    {
        m(i) /= rhs(i);
    }
    return m;
}

paz::ComplexMat paz::ComplexMatRef::operator*(const ComplexMatRef& rhs) const
{
    ComplexMat m = *this;
    m *= rhs;
    return m;
}

paz::ComplexMat paz::ComplexMatRef::operator+(const ComplexMatRef& rhs) const
{
    ComplexMat m = *this;
    m += rhs;
    return m;
}

paz::ComplexMat paz::ComplexMatRef::operator-(const ComplexMatRef& rhs) const
{
    ComplexMat m = *this;
    m -= rhs;
    return m;
}

paz::ComplexMat paz::ComplexMatRef::operator*(complex rhs) const
{
    ComplexMat m = *this;
    m *= rhs;
    return m;
}

paz::ComplexMat paz::ComplexMatRef::operator/(complex rhs) const
{
    ComplexMat m = *this;
    m /= rhs;
    return m;
}

paz::ComplexMat paz::ComplexMatRef::operator-() const
{
    return -ComplexMat(*this);
}

paz::complex paz::ComplexMatRef::dot(const ComplexMatRef& rhs) const
{
    if(rows() != rhs.rows() || cols() != rhs.cols())
    {
        throw std::runtime_error("Matrices must have the same dimensions.");
    }
    complex res = 0.;
    for(std::size_t i = 0; i < size(); ++i)
    {
        res += operator()(i)*rhs(i);
    }
    return res;
}

paz::ComplexVec paz::ComplexMatRef::cross(const ComplexMatRef& rhs) const
{
    return ComplexMat(*this).cross(rhs);
}

paz::ComplexMatRef paz::ComplexMatRef::block(std::size_t startRow, std::size_t
    startCol, std::size_t numRows, std::size_t numCols) const
{
    if(startRow + numRows > rows() || startCol + numCols > cols())
    {
        throw std::runtime_error("Block is out of range.");
    }
    return ComplexMatRef(_begin.ptr + startRow + _begin.origRows*startCol,
        _begin.origRows, _origCols, numRows, numCols);
}

paz::ComplexMatRef paz::ComplexMatRef::row(std::size_t m) const
{
    return block(m, 0, 1, cols());
}

paz::ComplexMatRef paz::ComplexMatRef::col(std::size_t n) const
{
    return block(0, n, rows(), 1);
}

bool paz::ComplexMatRef::hasNan() const
{
    for(auto n : *this)
    {
        if(std::isnan(n.real()) || std::isnan(n.imag()))
        {
            return true;
        }
    }
    return false;
}

paz::Mat paz::ComplexMatRef::real() const
{
    Mat m(rows(), cols());
    for(std::size_t i = 0; i < size(); ++i)
    {
        m._vals[i] = std::real(operator()(i));
    }
    return m;
}

paz::Mat paz::ComplexMatRef::imag() const
{
    Mat m(rows(), cols());
    for(std::size_t i = 0; i < size(); ++i)
    {
        m._vals[i] = std::imag(operator()(i));
    }
    return m;
}

paz::Mat paz::ComplexMatRef::abs() const
{
    Mat m(rows(), cols());
    for(std::size_t i = 0; i < size(); ++i)
    {
        m._vals[i] = std::abs(operator()(i));
    }
    return m;
}

paz::ComplexMat paz::ComplexMatRef::conj() const
{
    ComplexMat m(rows(), cols());
    for(std::size_t i = 0; i < size(); ++i)
    {
        m._vals[i] = std::conj(operator()(i));
    }
    return m;
}

paz::ComplexMat paz::ComplexMatRef::conjTrans() const
{
    return ComplexMat(*this).conjTrans();
}
