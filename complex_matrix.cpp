#include "PAZ_Math"
#define eigen_assert(x){ if(!(x)) throw std::runtime_error(#x); }
#include "Eigen"
#include <sstream>
#include <iomanip>

paz::ComplexMat paz::ComplexMat::Constant(std::size_t rows, std::size_t cols,
    complex c)
{
    ComplexMat m(rows, cols);
    std::fill(m.begin(), m.end(), c);
    return m;
}

paz::ComplexMat paz::ComplexMat::Constant(std::size_t side, complex c)
{
    return Constant(side, side, c);
}

paz::ComplexMat paz::ComplexMat::Zero(std::size_t rows, std::size_t cols)
{
    return Constant(rows, cols, 0.);
}

paz::ComplexMat paz::ComplexMat::Zero(std::size_t side)
{
    return Constant(side, side, 0.);
}

paz::ComplexMat paz::ComplexMat::Ones(std::size_t rows, std::size_t cols)
{
    return Constant(rows, cols, 1.);
}

paz::ComplexMat paz::ComplexMat::Ones(std::size_t side)
{
    return Constant(side, side, 1.);
}

paz::ComplexMat paz::ComplexMat::Identity(std::size_t side)
{
    ComplexMat m(side, side);
    for(std::size_t i = 0; i < side; ++i)
    {
        for(std::size_t j = 0; j < side; ++j)
        {
            m(i, j) = (i == j);
        }
    }
    return m;
}

paz::ComplexMat paz::ComplexMat::Diag(const ComplexMatRef& vals)
{
    ComplexMat m = ComplexMat::Zero(vals.size());
    for(std::size_t i = 0; i < vals.size(); ++i)
    {
        m(i, i) = vals(i);
    }
    return m;
}

paz::ComplexMat paz::ComplexMat::BlockDiag(const ComplexMatRef& a, const
    ComplexMatRef& b)
{
    ComplexMat m = ComplexMat::Zero(a.rows() + b.rows(), a.cols() + b.cols());
    m.setBlock(0, 0, a.rows(), a.cols(), a);
    m.setBlock(a.rows(), a.cols(), b.rows(), b.cols(), b);
    return m;
}

paz::ComplexMat paz::ComplexMat::Cross(const ComplexMatRef& vals)
{
    if(vals.rows() != 3 && vals.cols() != 1)
    {
        throw std::runtime_error("Not a 3-vector.");
    }
    return ComplexMat{{      0., -vals(2),  vals(1)},
               { vals(2),       0., -vals(0)},
               {-vals(1),  vals(0),       0.}};
}

paz::ComplexMat paz::ComplexMat::Hcat(const ComplexMatRef& a, const
    ComplexMatRef& b)
{
    if(a.rows() != b.rows())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    ComplexMat res(a.rows(), a.cols() + b.cols());
    std::copy(a.begin(), a.end(), res.begin());
    std::copy(b.begin(), b.end(), res.begin() + a.size());
    return res;
}

paz::ComplexMat paz::ComplexMat::Vcat(const ComplexMatRef& a, const
    ComplexMatRef& b)
{
    const std::size_t cols = a.cols();
    if(cols != b.cols())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    const std::size_t rows = a.rows() + b.rows();
    ComplexMat res(rows, cols);
    for(std::size_t i = 0; i < cols; ++i)
    {
        std::copy(a.begin() + a.rows()*i, a.begin() + a.rows()*(i + 1), res.
            begin() + rows*i);
        std::copy(b.begin() + b.rows()*i, b.begin() + b.rows()*(i + 1), res.
            begin() + rows*i + a.rows());
    }
    return res;
}

paz::ComplexMat paz::ComplexMat::Randn(std::size_t rows, std::size_t cols)
{
    ComplexMat m(rows, cols);
    for(auto& n : m)
    {
        n = randn();
    }
    return m;
}

paz::ComplexMat paz::ComplexMat::Randn(std::size_t side)
{
    return ComplexMat::Randn(side, side);
}

paz::ComplexMat::ComplexMat(std::size_t rows, std::size_t cols) : _vals(rows*
    cols), _rows(rows), _cols(cols) {}

paz::ComplexMat::ComplexMat(std::size_t side) : ComplexMat(side, side) {}

paz::ComplexMat::ComplexMat(const ComplexVec& v) : _vals(v._vals), _rows(v.
    rows()), _cols(1) {}

paz::ComplexMat::ComplexMat(const ComplexMatRef& m) : ComplexMat(m.rows(), m.
    cols())
{
    std::copy(m.begin(), m.end(), _vals.begin());
}

paz::ComplexMat::ComplexMat(const std::initializer_list<std::initializer_list<
    complex>>& list) : _rows(list.size())
{
    if(!_rows)
    {
        return;
    }
    _cols = list.begin()->size();
    _vals.resize(_rows*_cols);
    for(std::size_t i = 0; i < _rows; ++i)
    {
        if((list.begin() + i)->size() != _cols)
        {
            throw std::runtime_error("Matrix initializer list is malformed.");
        }
        for(std::size_t j = 0; j < _cols; ++j)
        {
            _vals[i + _rows*j] = *((list.begin() + i)->begin() + j);
        }
    }
}

paz::complex paz::ComplexMat::det() const
{
    if(rows() != cols() || empty())
    {
        throw std::runtime_error("Matrix must be square.");
    }
    Eigen::Map<const Eigen::MatrixXcd> m(data(), rows(), cols());
    return m.determinant();
}

paz::ComplexMat paz::ComplexMat::inv() const
{
    if(rows() != cols())
    {
        throw std::runtime_error("Matrix must be square.");
    }
    if(empty())
    {
        return *this;
    }
    ComplexMat res = *this;
    Eigen::Map<Eigen::MatrixXcd> m(res.data(), rows(), cols());
    m = m.inverse().eval();
    return res;
}

paz::ComplexMat paz::ComplexMat::solve(const ComplexMat& b) const //TEMP - not `ComplexMatRef` to support `Eigen::Map`
{
    if(rows() != b.rows())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    if(empty())
    {
        return *this;
    }
    Eigen::Map<const Eigen::MatrixXcd> eigenA(data(), rows(), cols());
    Eigen::Map<const Eigen::MatrixXcd> eigenB(b.data(), b.rows(), b.cols());
    ComplexMat x(cols(), b.cols());
    Eigen::Map<Eigen::MatrixXcd> eigenX(x.data(), x.rows(), x.cols());
    eigenX = eigenA.colPivHouseholderQr().solve(eigenB);
    return x;
}

paz::ComplexMat paz::ComplexMat::chol() const
{
    if(empty())
    {
        return *this;
    }
    if(rows() != cols())
    {
        throw std::runtime_error("Matrix must be square.");
    }
    Eigen::Map<const Eigen::MatrixXcd> m(data(), rows(), cols());
    if(m.hasNaN())
    {
        throw std::runtime_error("Matrix contains NaN.");
    }
    Eigen::LLT<Eigen::MatrixXcd> llt(m);
    if(llt.info() != Eigen::Success)
    {
        throw std::runtime_error("Cholesky decomposition failed.");
    }
    ComplexMat res(rows(), cols());
    Eigen::Map<Eigen::MatrixXcd> lMat(res.data(), rows(), cols());
    lMat = llt.matrixL();
    return res;
}

paz::ComplexMat paz::ComplexMat::cholUpdate(const ComplexMat& m, double a) const //TEMP - not `ComplexMatRef` to support `Eigen::Map`
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
    Eigen::Map<const Eigen::MatrixXcd> l(data(), rows(), cols());
    Eigen::Map<const Eigen::MatrixXcd> eigenM(m.data(), m.rows(), m.cols());
    if(l.hasNaN() || eigenM.hasNaN())
    {
        throw std::runtime_error("Matrix contains NaN.");
    }
    Eigen::LLT<Eigen::MatrixXcd> llt(Eigen::MatrixXcd{});
    const_cast<Eigen::MatrixXcd&>(llt.matrixLLT()) = l;
    for(std::size_t i = 0; i < m.cols(); ++i)
    {
        if(a < 0.)
        {
            llt.rankUpdate(eigenM.col(i), -std::sqrt(-a));
        }
        else
        {
            llt.rankUpdate(eigenM.col(i), std::sqrt(a));
        }
        if(llt.info() != Eigen::Success)
        {
            throw std::runtime_error("Cholesky update failed.");
        }
    }
    ComplexMat res(rows(), cols());
    Eigen::Map<Eigen::MatrixXcd> lMat(res.data(), rows(), cols());
    lMat = llt.matrixL();
    return res;
}

paz::ComplexVec paz::ComplexMat::eig() const
{
    if(empty())
    {
        return {};
    }
    if(rows() != cols())
    {
        throw std::runtime_error("Matrix must be square.");
    }
    Eigen::Map<const Eigen::MatrixXcd> m(data(), rows(), cols());
    if(m.hasNaN())
    {
        throw std::runtime_error("Matrix contains NaN.");
    }
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eig(m);
    if(eig.info() != Eigen::Success)
    {
        throw std::runtime_error("Eigendecomposition failed.");
    }
    ComplexVec vals(rows());
    for(std::size_t i = 0; i < rows(); ++i)
    {
        vals(i) = eig.eigenvalues()(i);
    }
    return vals;
}

paz::ComplexVec paz::ComplexMat::eig(ComplexMat& vecs) const
{
    if(empty())
    {
        vecs = {};
        return {};
    }
    if(rows() != cols())
    {
        throw std::runtime_error("Matrix must be square.");
    }
    Eigen::Map<const Eigen::MatrixXcd> m(data(), rows(), cols());
    if(m.hasNaN())
    {
        throw std::runtime_error("Matrix contains NaN.");
    }
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eig(m);
    if(eig.info() != Eigen::Success)
    {
        throw std::runtime_error("Eigendecomposition failed.");
    }
    ComplexVec vals(rows());
    for(std::size_t i = 0; i < rows(); ++i)
    {
        vals(i) = eig.eigenvalues()(i).imag() ? nan() : eig.eigenvalues()(i). //TEMP
            real();
    }
    vecs = ComplexMat(rows(), cols());
    for(std::size_t i = 0; i < rows(); ++i)
    {
        for(std::size_t j = 0; j < cols(); ++j)
        {
            vecs(i, j) = eig.eigenvectors()(i, j);
        }
    }
    return vals;
}

void paz::ComplexMat::qr(ComplexMat& q, ComplexMat& r) const //TEMP - not `ComplexMatRef` to support `Eigen::Map`
{
    if(empty())
    {
        q = {};
        r = {};
        return;
    }
    Eigen::Map<const Eigen::MatrixXcd> m(data(), rows(), cols());
    if(m.hasNaN())
    {
        throw std::runtime_error("Matrix contains NaN.");
    }
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(m);
    q.resize(rows(), rows());
    Eigen::Map<Eigen::MatrixXcd> eigenQ(q.data(), q.rows(), q.cols());
    eigenQ = qr.householderQ();
    r.resize(rows(), cols());
    Eigen::Map<Eigen::MatrixXcd> eigenR(r.data(), r.rows(), r.cols());
    eigenR = qr.matrixQR().triangularView<Eigen::Upper>();
}

void paz::ComplexMat::qr(ComplexMat& q, ComplexMat& r, std::vector<std::size_t>&
    p) const //TEMP - not `ComplexMatRef` to support `Eigen::Map`
{
    if(empty())
    {
        q = {};
        r = {};
        p = {};
        return;
    }
    Eigen::Map<const Eigen::MatrixXcd> m(data(), rows(), cols());
    if(m.hasNaN())
    {
        throw std::runtime_error("Matrix contains NaN.");
    }
    Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> qr(m);
    q.resize(rows(), rows());
    Eigen::Map<Eigen::MatrixXcd> eigenQ(q.data(), q.rows(), q.cols());
    eigenQ = qr.householderQ();
    r.resize(rows(), cols());
    Eigen::Map<Eigen::MatrixXcd> eigenR(r.data(), r.rows(), r.cols());
    eigenR = qr.matrixQR().triangularView<Eigen::Upper>();
    p.resize(cols());
    std::copy(qr.colsPermutation().indices().begin(), qr.colsPermutation().
        indices().end(), p.begin());
}

paz::ComplexMat paz::ComplexMat::trans() const
{
    if(empty())
    {
        return *this;
    }
    if(rows() == 1)
    {
        auto res = *this;
        res._rows = res.size();
        res._cols = 1;
        return res;
    }
    if(cols() == 1)
    {
        auto res = *this;
        res._rows = 1;
        res._cols = res.size();
        return res;
    }
    ComplexMat res(cols(), rows());
    for(std::size_t i = 0; i < _rows; ++i)
    {
        for(std::size_t j = 0; j < res._rows; ++j)
        {
            res._vals[j + res._rows*i] = _vals[i + _rows*j];
        }
    }
    return res;
}

paz::ComplexVec paz::ComplexMat::diag() const
{
    if(rows() != cols() || empty())
    {
        throw std::runtime_error("Matrix must be square.");
    }
    ComplexVec res(_rows);
    for(std::size_t i = 0; i < _rows; ++i)
    {
        res(i) = _vals[i + _rows*i];
    }
    return res;
}

paz::ComplexMat paz::ComplexMat::rep(std::size_t m, std::size_t n) const
{
    ComplexMat res(m*rows(), n*cols());
    for(std::size_t i = 0; i < m*n; ++i)
    {
        std::copy(begin(), end(), res.begin() + rows()*cols()*i);
    }
    return res;
}

paz::complex paz::ComplexMat::normSq() const
{
    return dot(*this);
}

paz::complex paz::ComplexMat::norm() const
{
    return std::sqrt(normSq());
}

paz::complex paz::ComplexMat::sum() const
{
    return std::accumulate(begin(), end(), complex{0.});
}

paz::ComplexVec paz::ComplexMat::rowSum() const
{
    ComplexVec res = ComplexVec::Zero(rows());
    for(std::size_t i = 0; i < rows(); ++i)
    {
        for(std::size_t j = 0; j < cols(); ++j)
        {
            res(i) += _vals[i + _rows*j];
        }
    }
    return res;
}

paz::ComplexMat paz::ComplexMat::colSum() const
{
    ComplexMat res = ComplexMat::Zero(1, cols());
    for(std::size_t i = 0; i < cols(); ++i)
    {
        for(std::size_t j = 0; j < rows(); ++j)
        {
            res(0, i) += _vals[j + _rows*i];
        }
    }
    return res;
}

paz::complex paz::ComplexMat::min() const
{
    return *std::min_element(begin(), end(), [](complex a, complex b){ return
        std::abs(a) < std::abs(b); });
}

paz::complex paz::ComplexMat::max() const
{
    return *std::max_element(begin(), end(), [](complex a, complex b){ return
        std::abs(a) > std::abs(b); });
}

paz::ComplexMat paz::ComplexMat::normalized() const
{
    return (*this)/norm();
}

paz::ComplexMat paz::ComplexMat::prod(const ComplexMatRef& rhs) const
{
    if(rows() != rhs.rows() || cols() != rhs.cols())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    auto res = *this;
    for(std::size_t i = 0; i < size(); ++i)
    {
        res(i) *= rhs(i);
    }
    return res;
}

paz::ComplexMat paz::ComplexMat::quot(const ComplexMatRef& rhs) const
{
    if(rows() != rhs.rows() || cols() != rhs.cols())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    auto res = *this;
    for(std::size_t i = 0; i < size(); ++i)
    {
        res(i) /= rhs(i);
    }
    return res;
}

paz::ComplexMat& paz::ComplexMat::operator*=(const ComplexMatRef& rhs)
{
    return *this = (*this)*rhs;
}

paz::ComplexMat paz::ComplexMat::operator*(const ComplexMatRef& rhs) const
{
    if(cols() != rhs.rows())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    ComplexMat res(rows(), rhs.cols());
    std::fill(res.begin(), res.end(), 0.);
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

paz::ComplexMat& paz::ComplexMat::operator+=(const ComplexMatRef& rhs)
{
    if(rows() != rhs.rows() || cols() != rhs.cols())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    for(std::size_t i = 0; i < size(); ++i)
    {
        _vals[i] += rhs(i);
    }
    return *this;
}

paz::ComplexMat paz::ComplexMat::operator+(const ComplexMatRef& rhs) const
{
    auto res = *this;
    return res += rhs;
}

paz::ComplexMat& paz::ComplexMat::operator-=(const ComplexMatRef& rhs)
{
    if(rows() != rhs.rows() || cols() != rhs.cols())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    for(std::size_t i = 0; i < size(); ++i)
    {
        _vals[i] -= rhs(i);
    }
    return *this;
}

paz::complex paz::ComplexMat::dot(const ComplexMatRef& rhs) const
{
    if(rows() != rhs.rows() || cols() != rhs.cols())
    {
        throw std::runtime_error("Matrices must have the same dimensions.");
    }
    complex res = 0.;
    for(std::size_t i = 0; i < size(); ++i)
    {
        res += _vals[i]*rhs(i);
    }
    return res;
}

paz::ComplexVec paz::ComplexMat::cross(const ComplexMatRef& rhs) const
{
    if(rows() != 3 || cols() != 1 || rhs.rows() != 3 || rhs.cols() != 1)
    {
        throw std::runtime_error("Not a 3-vector.");
    }
    return {{operator()(1)*rhs(2) - operator()(2)*rhs(1),
             operator()(2)*rhs(0) - operator()(0)*rhs(2),
             operator()(0)*rhs(1) - operator()(1)*rhs(0)}};
}

paz::ComplexMat paz::ComplexMat::operator-(const ComplexMatRef& rhs) const
{
    auto res = *this;
    return res -= rhs;
}

paz::ComplexMat& paz::ComplexMat::operator*=(complex rhs)
{
    for(auto& n : _vals)
    {
        n *= rhs;
    }
    return *this;
}

paz::ComplexMat paz::ComplexMat::operator*(complex rhs) const
{
    auto res = *this;
    return res *= rhs;
}

paz::ComplexMat& paz::ComplexMat::operator/=(complex rhs)
{
    for(auto& n : _vals)
    {
        n /= rhs;
    }
    return *this;
}

paz::ComplexMat paz::ComplexMat::operator/(complex rhs) const
{
    auto res = *this;
    return res /= rhs;
}

paz::ComplexMat paz::ComplexMat::operator-() const
{
    auto res = *this;
    for(auto& n : res._vals)
    {
        n = -n;
    }
    return res;
}

paz::ComplexMatRef paz::ComplexMat::block(std::size_t startRow, std::size_t
    startCol, std::size_t numRows, std::size_t numCols) const
{
    if(startRow + numRows > rows() || startCol + numCols > cols())
    {
        throw std::runtime_error("Block is out of range.");
    }
    return ComplexMatRef(data() + startRow + rows()*startCol, rows(), cols(),
        numRows, numCols);
}

void paz::ComplexMat::setBlock(std::size_t startRow, std::size_t startCol, std::
    size_t numRows, std::size_t numCols, const ComplexMatRef& rhs)
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
            _vals[startRow + i + _rows*(startCol + j)] = rhs(i, j);
        }
    }
}

paz::ComplexMatRef paz::ComplexMat::row(std::size_t m) const
{
    return block(m, 0, 1, _cols);
}

void paz::ComplexMat::setRow(std::size_t m, const ComplexMatRef& rhs)
{
    if(rhs.rows() != 1 || rhs.cols() != cols())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    for(std::size_t i = 0; i < _cols; ++i)
    {
        _vals[m + _rows*i] = rhs(i);
    }
}

paz::ComplexMatRef paz::ComplexMat::col(std::size_t n) const
{
    return block(0, n, _rows, 1);
}

void paz::ComplexMat::setCol(std::size_t n, const ComplexMatRef& rhs)
{
    if(rhs.rows() != rows() || rhs.cols() != 1)
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    for(std::size_t i = 0; i < _rows; ++i)
    {
        _vals[i + _rows*n] = rhs(i);
    }
}

void paz::ComplexMat::resize(std::size_t newRows, std::size_t newCols)
{
    resizeRows(newRows);
    resizeCols(newCols);
}

void paz::ComplexMat::resize(std::size_t newRows, std::size_t newCols, complex
    c)
{
    resizeRows(newRows, c);
    resizeCols(newCols, c);
}

void paz::ComplexMat::resizeRows(std::size_t newRows)
{
    if(newRows == _rows)
    {
        return;
    }
    if(empty())
    {
        _vals.resize(newRows*_cols);
    }
    else
    {
        std::vector<complex> newVals(newRows*_cols);
        const std::size_t copyRows = std::min(newRows, _rows);
        for(std::size_t i = 0; i < _cols; ++i)
        {
            std::copy(begin() + _rows*i, begin() + _rows*i + copyRows, newVals.
                begin() + newRows*i);
        }
        std::swap(newVals, _vals);
    }
    _rows = newRows;
}

void paz::ComplexMat::resizeRows(std::size_t newRows, complex c)
{
    if(newRows == _rows)
    {
        return;
    }
    if(empty())
    {
        _vals.resize(newRows*_cols, c);
    }
    else
    {
        std::vector<complex> newVals(newRows*_cols, c);
        const std::size_t copyRows = std::min(newRows, _rows);
        for(std::size_t i = 0; i < _cols; ++i)
        {
            std::copy(begin() + _rows*i, begin() + _rows*i + copyRows, newVals.
                begin() + newRows*i);
        }
        std::swap(newVals, _vals);
    }
    _rows = newRows;
}

void paz::ComplexMat::resizeCols(std::size_t newCols)
{
    if(newCols == _cols)
    {
        return;
    }
    _vals.resize(_rows*newCols);
    _cols = newCols;
}

void paz::ComplexMat::resizeCols(std::size_t newCols, complex c)
{
    if(newCols == _cols)
    {
        return;
    }
    _vals.resize(_rows*newCols, c);
    _cols = newCols;
}

bool paz::ComplexMat::hasNan() const
{
    for(auto n : _vals)
    {
        if(std::isnan(n.real()) || std::isnan(n.imag()))
        {
            return true;
        }
    }
    return false;
}

void paz::ComplexMat::shuffleCols()
{
    if(!rows() || cols() < 2)
    {
        return;
    }
    const auto seq = rand_seq(cols());
    std::vector<complex> newVals(rows()*cols());
    for(std::size_t i = 0; i < cols(); ++i)
    {
        std::copy(begin() + _rows*i, begin() + _rows*(i + 1), newVals.begin() +
            _rows*seq[i]);
    }
    swap(newVals, _vals);
}

paz::Mat paz::ComplexMat::real() const
{
    Mat m(_rows, _cols);
    for(std::size_t i = 0; i < _vals.size(); ++i)
    {
        m._vals[i] = _vals[i].real();
    }
    return m;
}

paz::Mat paz::ComplexMat::imag() const
{
    Mat m(_rows, _cols);
    for(std::size_t i = 0; i < _vals.size(); ++i)
    {
        m._vals[i] = _vals[i].imag();
    }
    return m;
}

paz::Mat paz::ComplexMat::abs() const
{
    Mat m(_rows, _cols);
    for(std::size_t i = 0; i < _vals.size(); ++i)
    {
        m._vals[i] = std::abs(_vals[i]);
    }
    return m;
}

paz::Mat paz::ComplexMat::arg() const
{
    Mat m(_rows, _cols);
    for(std::size_t i = 0; i < _vals.size(); ++i)
    {
        m._vals[i] = std::arg(_vals[i]);
    }
    return m;
}

paz::ComplexMat paz::ComplexMat::conj() const
{
    ComplexMat m(_rows, _cols);
    for(std::size_t i = 0; i < _vals.size(); ++i)
    {
        m._vals[i] = std::conj(_vals[i]);
    }
    return m;
}

paz::ComplexMat paz::ComplexMat::conjTrans() const
{
    if(empty())
    {
        return *this;
    }
    if(rows() == 1)
    {
        auto res = conj();
        res._rows = res.size();
        res._cols = 1;
        return res;
    }
    if(cols() == 1)
    {
        auto res = conj();
        res._rows = 1;
        res._cols = res.size();
        return res;
    }
    ComplexMat res(cols(), rows());
    for(std::size_t i = 0; i < _rows; ++i)
    {
        for(std::size_t j = 0; j < res._rows; ++j)
        {
            res._vals[j + res._rows*i] = std::conj(_vals[i + _rows*j]);
        }
    }
    return res;
}

std::ostream& paz::operator<<(std::ostream& out, const ComplexMatRef& rhs)
{
    if(!rhs.empty())
    {
        std::vector<std::string> str;
        str.reserve(rhs.rows()*rhs.cols());
        std::size_t maxLen = 0;
        for(std::size_t i = 0; i < rhs.rows(); ++i)
        {
            for(std::size_t j = 0; j < rhs.cols(); ++j)
            {
                std::ostringstream oss;
                oss.flags(out.flags());
                oss.precision(out.precision());
                oss << rhs(i, j);
                str.push_back(oss.str());
                maxLen = std::max(maxLen, str.back().size());
            }
        }
        if(out.width())
        {
            maxLen = out.width();
        }
        for(std::size_t i = 0; i < rhs.rows(); ++i)
        {
            for(std::size_t j = 0; j < rhs.cols(); ++j)
            {
                out << std::setw(maxLen) << str[rhs.cols()*i + j];
                if(j + 1 < rhs.cols())
                {
                    out << ' ';
                }
            }
            if(i + 1 < rhs.rows())
            {
                out << '\n';
            }
        }
    }
    return out;
}
