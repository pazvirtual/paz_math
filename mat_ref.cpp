#include "PAZ_Math"


paz::MatRef::MatRef(const double* ptr, std::size_t origRows, std::size_t
    origCols, std::size_t blockRows, std::size_t blockCols) : _begin({ptr, 0,
    static_cast<iterator::difference_type>(origRows), static_cast<iterator::
    difference_type>(blockRows)}), _origCols(origCols), _blockCols(blockCols) {}

paz::MatRef::MatRef(const Mat& m) : MatRef(m.data(), m.rows(), m.cols(), m.
    rows(), m.cols()) {}

double paz::MatRef::det() const
{
    return Mat(*this).det();
}

paz::Mat paz::MatRef::inv() const
{
    return Mat(*this).inv();
}

paz::Mat paz::MatRef::solve(const Mat& b) const
{
    return Mat(*this).solve(b);
}

paz::Mat paz::MatRef::chol() const
{
    return Mat(*this).chol();
}

paz::Mat paz::MatRef::cholUpdate(const Mat& m, double a) const
{
    return Mat(*this).cholUpdate(m, a);
}

paz::Vec paz::MatRef::eig() const
{
    return Mat(*this).eig();
}

paz::Vec paz::MatRef::eig(Mat& vecs) const
{
    return Mat(*this).eig(vecs);
}

void paz::MatRef::qr(Mat& q, Mat& r) const
{
    Mat(*this).qr(q, r);
}

void paz::MatRef::qr(Mat& q, Mat& r, std::vector<std::size_t>& p) const
{
    Mat(*this).qr(q, r, p);
}

paz::Mat paz::MatRef::trans() const
{
    return Mat(*this).trans();
}

paz::Vec paz::MatRef::diag() const
{
    if(rows() != cols() || empty())
    {
        throw std::runtime_error("Matrix must be square.");
    }
    Vec res(rows());
    for(std::size_t i = 0; i < rows(); ++i)
    {
        res(i) = operator()(i, i);
    }
    return res;
}

paz::Mat paz::MatRef::rep(std::size_t m, std::size_t n) const
{
    Mat res(m*rows(), n*cols());
    for(std::size_t i = 0; i < m*n; ++i)
    {
        std::copy(begin(), end(), res.begin() + rows()*cols()*i);
    }
    return res;
}

double paz::MatRef::normSq() const
{
    return dot(*this);
}

double paz::MatRef::norm() const
{
    return std::sqrt(normSq());
}

double paz::MatRef::sum() const
{
    return std::accumulate(begin(), end(), 0.);
}

paz::Vec paz::MatRef::rowSum() const
{
    Vec res = Vec::Zero(rows());
    for(std::size_t i = 0; i < rows(); ++i)
    {
        for(std::size_t j = 0; j < cols(); ++j)
        {
            res(i) += operator()(i, j);
        }
    }
    return res;
}

paz::Mat paz::MatRef::colSum() const
{
    Mat res = Mat::Zero(1, cols());
    for(std::size_t i = 0; i < cols(); ++i)
    {
        for(std::size_t j = 0; j < rows(); ++j)
        {
            res(0, i) += operator()(j, i);
        }
    }
    return res;
}

double paz::MatRef::min() const
{
    return *std::min_element(begin(), end());
}

double paz::MatRef::max() const
{
    return *std::max_element(begin(), end());
}

paz::Mat paz::MatRef::normalized() const
{
    Mat m = *this;
    m /= norm();
    return m;
}

paz::Mat paz::MatRef::prod(const MatRef& rhs) const // elementwise
{
    if(rows() != rhs.rows() || cols() != rhs.cols())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    Mat m = *this;
    for(std::size_t i = 0; i < size(); ++i)
    {
        m(i) *= rhs(i);
    }
    return m;
}

paz::Mat paz::MatRef::quot(const MatRef& rhs) const // elementwise
{
    if(rows() != rhs.rows() || cols() != rhs.cols())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    Mat m = *this;
    for(std::size_t i = 0; i < size(); ++i)
    {
        m(i) /= rhs(i);
    }
    return m;
}

paz::Mat paz::MatRef::operator*(const MatRef& rhs) const
{
    Mat m = *this;
    m *= rhs;
    return m;
}

paz::Mat paz::MatRef::operator+(const MatRef& rhs) const
{
    Mat m = *this;
    m += rhs;
    return m;
}

paz::Mat paz::MatRef::operator-(const MatRef& rhs) const
{
    Mat m = *this;
    m -= rhs;
    return m;
}

paz::Mat paz::MatRef::operator*(double rhs) const
{
    Mat m = *this;
    m *= rhs;
    return m;
}

paz::Mat paz::MatRef::operator/(double rhs) const
{
    Mat m = *this;
    m /= rhs;
    return m;
}

paz::Mat paz::MatRef::operator-() const
{
    return -Mat(*this);
}

double paz::MatRef::dot(const MatRef& rhs) const
{
    if(rows() != rhs.rows() || cols() != rhs.cols())
    {
        throw std::runtime_error("Matrices must have the same dimensions.");
    }
    double res = 0.;
    for(std::size_t i = 0; i < size(); ++i)
    {
        res += operator()(i)*rhs(i);
    }
    return res;
}

paz::Vec paz::MatRef::cross(const MatRef& rhs) const
{
    return Mat(*this).cross(rhs);
}

paz::MatRef paz::MatRef::block(std::size_t startRow, std::size_t startCol, std::
    size_t numRows, std::size_t numCols) const
{
    if(startRow + numRows > rows() || startCol + numCols > cols())
    {
        throw std::runtime_error("Block is out of range.");
    }
    return MatRef(_begin.ptr + startRow + _begin.origRows*startCol, _begin.
        origRows, _origCols, numRows, numCols);
}

paz::MatRef paz::MatRef::row(std::size_t m) const
{
    return block(m, 0, 1, cols());
}

paz::MatRef paz::MatRef::col(std::size_t n) const
{
    return block(0, n, rows(), 1);
}

bool paz::MatRef::hasNan() const
{
    for(auto n : *this)
    {
        if(std::isnan(n))
        {
            return true;
        }
    }
    return false;
}
