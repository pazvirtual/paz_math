#include "PAZ_Math"
#define eigen_assert(X) { if(!(X)) throw std::runtime_error(#X); }
#include "Eigen"

paz::Mat paz::Mat::Constant(std::size_t rows, std::size_t cols, double c)
{
    paz::Mat m(rows, cols);
    std::fill(m.begin(), m.end(), c);
    return m;
}

paz::Mat paz::Mat::Constant(std::size_t side, double c)
{
    return Constant(side, side, c);
}

paz::Mat paz::Mat::Zero(std::size_t rows, std::size_t cols)
{
    return Constant(rows, cols, 0.);
}

paz::Mat paz::Mat::Zero(std::size_t side)
{
    return Constant(side, side, 0.);
}

paz::Mat paz::Mat::Ones(std::size_t rows, std::size_t cols)
{
    return Constant(rows, cols, 1.);
}

paz::Mat paz::Mat::Ones(std::size_t side)
{
    return Constant(side, side, 1.);
}

paz::Mat paz::Mat::Identity(std::size_t side)
{
    Mat m(side, side);
    for(std::size_t i = 0; i < side; ++i)
    {
        for(std::size_t j = 0; j < side; ++j)
        {
            m._vals[i + m._rows*j] = (i == j);
        }
    }
    return m;
}

paz::Mat::Mat(std::size_t rows, std::size_t cols) : _vals(rows*cols), _rows(
    rows) {}

paz::Mat::Mat(std::size_t side) : Mat(side, side) {}

paz::Mat::Mat(const Vec& v) : _vals(v._vals), _rows(v._rows) {}

paz::Mat::Mat(const std::initializer_list<std::initializer_list<double>>& list)
    : _rows(list.size())
{
    if(!_rows)
    {
        return;
    }
    const std::size_t cols = list.begin()->size();
    _vals.resize(_rows*cols);
    for(std::size_t i = 0; i < _rows; ++i)
    {
        if((list.begin() + i)->size() != cols)
        {
            throw std::runtime_error("Matrix initializer list is malformed.");
        }
        for(std::size_t j = 0; j < cols; ++j)
        {
            _vals[i + _rows*j] = *((list.begin() + i)->begin() + j);
        }
    }
}

paz::Mat paz::Mat::inv() const
{
    if(rows() != cols())
    {
        throw std::runtime_error("Matrix must be square.");
    }
    if(empty())
    {
        return *this;
    }
    Eigen::MatrixXd m(rows(), cols());
    std::copy(begin(), end(), m.data());
    m = m.inverse().eval();
    Mat res(rows(), cols());
    std::copy(m.data(), m.data() + m.size(), res.begin());
    return res;
}

paz::Mat paz::Mat::solve(const Mat& b) const
{
    if(rows() != cols())
    {
        throw std::runtime_error("Matrix must be square.");
    }
    if(cols() != b.rows())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    if(empty())
    {
        return *this;
    }
    Eigen::MatrixXd eigenA(rows(), cols());
    std::copy(begin(), end(), eigenA.data());
    Eigen::MatrixXd eigenB(b.rows(), b.cols());
    std::copy(b.begin(), b.end(), eigenB.data());
    const Eigen::MatrixXd eigenX = eigenA.colPivHouseholderQr().solve(eigenB);
    Mat res(eigenX.rows(), eigenX.cols());
    std::copy(eigenX.data(), eigenX.data() + eigenX.size(), res.begin());
    return res;
}

paz::Mat paz::Mat::chol() const
{
    if(empty())
    {
        return *this;
    }
    if(rows() != cols())
    {
        throw std::runtime_error("Matrix must be square.");
    }
    Eigen::MatrixXd m(rows(), cols());
    std::copy(begin(), end(), m.data());
    if(m.hasNaN())
    {
        throw std::runtime_error("Matrix contains NaN.");
    }
    Eigen::LLT<Eigen::MatrixXd> llt(m);
    if(llt.info() == Eigen::NumericalIssue)
    {
        throw std::runtime_error("Cholesky decomposition failed.");
    }
    const Eigen::MatrixXd lMat = llt.matrixL();
    Mat res(rows(), cols());
    std::copy(lMat.data(), lMat.data() + lMat.size(), res.begin());
    return res;
}

paz::Mat paz::Mat::cholUpdate(const Mat& m, double a) const
{
    if(empty())
    {
        return *this;
    }
    if(rows() != cols())
    {
        throw std::logic_error("Matrix must be square.");
    }
    if(rows() != m.rows())
    {
        throw std::logic_error("Matrices must have the same number of rows.");
    }
    Eigen::MatrixXd l(rows(), cols());
    std::copy(begin(), end(), l.data());
    Eigen::MatrixXd eigenM(m.rows(), m.cols());
    std::copy(m.begin(), m.end(), eigenM.data());
    if(l.hasNaN() || eigenM.hasNaN())
    {
        throw std::runtime_error("Matrix contains NaN.");
    }
    Eigen::LLT<Eigen::MatrixXd> llt(Eigen::MatrixXd{});
    const_cast<Eigen::MatrixXd&>(llt.matrixLLT()) = l;
    for(std::size_t i = 0; i < m.cols(); ++i)
    {
        llt.rankUpdate(eigenM.col(i), a);
        if(llt.info() == Eigen::NumericalIssue)
        {
            throw std::runtime_error("Cholesky update failed.");
        }
    }
    l = llt.matrixL();
    Mat res(rows(), cols());
    std::copy(l.data(), l.data() + l.size(), res.begin());
    return res;
}

paz::Mat paz::Mat::trans() const
{
    if(_rows == 1)
    {
        auto res = *this;
        res._rows = res.size();
        return res;
    }
    if(cols() == 1)
    {
        auto res = *this;
        res._rows = 1;
        return res;
    }
    Mat res(cols(), rows());
    for(std::size_t i = 0; i < _rows; ++i)
    {
        for(std::size_t j = 0; j < res._rows; ++j)
        {
            res._vals[j + res._rows*i] = _vals[i + _rows*j];
        }
    }
    return res;
}

double& paz::Mat::operator()(std::size_t i, std::size_t j)
{
    if(i >= rows())
    {
        throw std::runtime_error("Row index out of range.");
    }
    if(j >= cols())
    {
        throw std::runtime_error("Column index out of range.");
    }
    return _vals[i + _rows*j];
}

double paz::Mat::operator()(std::size_t i, std::size_t j) const
{
    if(i >= rows())
    {
        throw std::runtime_error("Row index out of range.");
    }
    if(j >= cols())
    {
        throw std::runtime_error("Column index out of range.");
    }
    return _vals[i + _rows*j];
}

double& paz::Mat::operator()(std::size_t i)
{
    if(i >= _vals.size())
    {
        throw std::runtime_error("Index is out of range.");
    }
    return _vals[i];
}

double paz::Mat::operator()(std::size_t i) const
{
    if(i >= _vals.size())
    {
        throw std::runtime_error("Index is out of range.");
    }
    return _vals[i];
}

double* paz::Mat::data()
{
    return _vals.data();
}

const double* paz::Mat::data() const
{
    return _vals.data();
}

bool paz::Mat::empty() const
{
    return _vals.empty();
}

std::size_t paz::Mat::size() const
{
    return _vals.size();
}

std::size_t paz::Mat::rows() const
{
    return _vals.size() ? _rows : 0;
}

std::size_t paz::Mat::cols() const
{
    return _vals.size() ? _vals.size()/_rows : 0;
}

double paz::Mat::normSq() const
{
    return dot(*this);
}

double paz::Mat::norm() const
{
    return std::sqrt(normSq());
}

paz::Mat paz::Mat::normalized() const
{
    return (*this)/norm();
}

paz::Mat& paz::Mat::operator*=(const Mat& rhs)
{
    return *this = (*this)*rhs;
}

paz::Mat paz::Mat::operator*(const Mat& rhs) const
{
    if(cols() != rhs.rows())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    Mat res(rows(), rhs.cols());
    std::fill(res._vals.begin(), res._vals.end(), 0.);
    for(std::size_t i = 0; i < rows(); ++i)
    {
        for(std::size_t j = 0; j < cols(); ++j)
        {
            for(std::size_t k = 0; k < rhs.cols(); ++k)
            {
                res(i, k) += _vals[i + _rows*j]*rhs(j, k);
            }
        }
    }
    return res;
}

paz::Mat& paz::Mat::operator+=(const Mat& rhs)
{
    if(rows() != rhs.rows() || cols() != rhs.cols())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    for(std::size_t i = 0; i < size(); ++i)
    {
        _vals[i] += rhs._vals[i];
    }
    return *this;
}

paz::Mat paz::Mat::operator+(const Mat& rhs) const
{
    auto res = *this;
    return res += rhs;
}

paz::Mat& paz::Mat::operator-=(const Mat& rhs)
{
    if(rows() != rhs.rows() || cols() != rhs.cols())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    for(std::size_t i = 0; i < size(); ++i)
    {
        _vals[i] -= rhs._vals[i];
    }
    return *this;
}

double paz::Mat::dot(const Mat& rhs) const
{
    if(size() != rhs.size())
    {
        throw std::runtime_error("Matrices must have the same size.");
    }
    double res = 0.;
    for(std::size_t i = 0; i < size(); ++i)
    {
        res += _vals[i]*rhs._vals[i];
    }
    return res;
}

paz::Mat paz::Mat::operator-(const Mat& rhs) const
{
    auto res = *this;
    return res -= rhs;
}

paz::Mat& paz::Mat::operator*=(double rhs)
{
    for(auto& n : _vals)
    {
        n *= rhs;
    }
    return *this;
}

paz::Mat paz::Mat::operator*(double rhs) const
{
    auto res = *this;
    return res *= rhs;
}

paz::Mat& paz::Mat::operator/=(double rhs)
{
    for(auto& n : _vals)
    {
        n /= rhs;
    }
    return *this;
}

paz::Mat paz::Mat::operator/(double rhs) const
{
    auto res = *this;
    return res /= rhs;
}

paz::Mat paz::Mat::operator-() const
{
    auto res = *this;
    for(auto& n : res._vals)
    {
        n = -n;
    }
    return res;
}

paz::Mat paz::Mat::block(std::size_t startRow, std::size_t startCol, std::
    size_t numRows, std::size_t numCols) const
{
    if(startRow + numRows > rows() || startCol + numCols > cols())
    {
        throw std::runtime_error("Block is out of range.");
    }
    paz::Mat res(numRows, numCols);
    for(std::size_t i = 0; i < numRows; ++i)
    {
        for(std::size_t j = 0; j < numCols; ++j)
        {
            res._vals[i + res._rows*j] = _vals[startRow + i + _rows*(startCol +
                j)];
        }
    }
    return res;
}

void paz::Mat::setBlock(std::size_t startRow, std::size_t startCol, std::
    size_t numRows, std::size_t numCols, const Mat& rhs)
{
    if(startRow + numRows > rows() || startCol + numCols > cols())
    {
        throw std::runtime_error("Block is out of range.");
    }
    if(rhs.rows() != numRows || rhs.cols() != numCols)
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    for(std::size_t i = 0; i < numRows; ++i)
    {
        for(std::size_t j = 0; j < numCols; ++j)
        {
            _vals[startRow + i + _rows*(startCol + j)] = rhs._vals[i + rhs.
                _rows*j];
        }
    }
}

paz::Mat paz::Mat::row(std::size_t m) const
{
    return block(m, 0, 1, cols());
}

void paz::Mat::setRow(std::size_t m, const Mat& rhs)
{
    if(rhs.rows() != 1 || rhs.cols() != cols())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    for(std::size_t i = 0; i < _rows; ++i)
    {
        _vals[m + _rows*i] = rhs._vals[i];
    }
}

paz::Mat paz::Mat::col(std::size_t n) const
{
    return block(0, n, rows(), 1);
}

void paz::Mat::setCol(std::size_t n, const Mat& rhs)
{
    if(rhs.rows() != rows() || rhs.cols() != 1)
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    for(std::size_t i = 0; i < _rows; ++i)
    {
        _vals[i + _rows*n] = rhs._vals[i];
    }
}

paz::Mat& paz::operator*=(double lhs, Mat& rhs)
{
    return rhs *= lhs;
}

paz::Mat paz::operator*(double lhs, const Mat& rhs)
{
    return rhs*lhs;
}

std::ostream& paz::operator<<(std::ostream& out, const Mat& rhs)
{
    for(std::size_t i = 0; i < rhs.rows(); ++i)
    {
        for(std::size_t j = 0; j < rhs.cols(); ++j)
        {
            out << rhs(i, j) << (j + 1 < rhs.cols() ? " " : "");
        }
        out << (i + 1 < rhs.rows() ? "\n" : "");
    }
    return out;
}
