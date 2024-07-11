#include "PAZ_Math"

paz::MatRef::iterator& paz::MatRef::iterator::operator--()
{
    --row;
    if(row < 0)
    {
        ptr += blockRows;
        ptr -= origRows + 1;
        row = blockRows - 1;
    }
    else
    {
        --ptr;
    }
    return *this;
}

paz::MatRef::iterator paz::MatRef::iterator::operator--(int)
{
    auto temp = *this;
    --(*this);
    return temp;
}

paz::MatRef::iterator& paz::MatRef::iterator::operator++()
{
    ++row;
    if(row == static_cast<difference_type>(blockRows))
    {
        ptr += origRows + 1;
        ptr -= blockRows;
        row = 0;
    }
    else
    {
        ++ptr;
    }
    return *this;
}

paz::MatRef::iterator paz::MatRef::iterator::operator++(int)
{
    auto temp = *this;
    ++(*this);
    return temp;
}

paz::MatRef::iterator& paz::MatRef::iterator::operator-=(difference_type n)
{
    return *this += -n;
}

paz::MatRef::iterator& paz::MatRef::iterator::operator+=(difference_type n)
{
    const difference_type temp = row + n;
    difference_type deltaCol = temp/blockRows;
    difference_type deltaRow = n - blockRows*deltaCol;
    if(temp < 0 && temp%blockRows)
    {
        --deltaCol;
        deltaRow += blockRows;
    }
    ptr += deltaRow + origRows*deltaCol;
    row += deltaRow;
    return *this;
}

paz::MatRef::iterator paz::MatRef::iterator::operator-(difference_type n) const
{
    auto temp = *this;
    temp -= n;
    return temp;
}

paz::MatRef::iterator paz::MatRef::iterator::operator+(difference_type n) const
{
    auto temp = *this;
    temp += n;
    return temp;
}

paz::MatRef::iterator::difference_type paz::MatRef::iterator::operator-(const
    iterator& it) const
{
    difference_type deltaPtr;
    if(ptr < it.ptr)
    {
        deltaPtr = -static_cast<difference_type>(it.ptr - ptr);
    }
    else
    {
        deltaPtr = ptr - it.ptr;
    }
    difference_type deltaCol = deltaPtr/origRows;
    if(deltaPtr < 0 && deltaPtr%origRows)
    {
        --deltaCol;
    }
    const difference_type deltaRow = row - it.row;
    return deltaRow + blockRows*deltaCol;
}

paz::MatRef::iterator::reference paz::MatRef::iterator::operator*() const
{
    return *ptr;
}

paz::MatRef::iterator::reference paz::MatRef::iterator::operator[](
    difference_type n) const
{
    return *(*this + n);
}

bool paz::MatRef::iterator::operator==(const iterator& it) const
{
    return ptr == it.ptr;
}

bool paz::MatRef::iterator::operator!=(const iterator& it) const
{
    return ptr != it.ptr;
}

bool paz::MatRef::iterator::operator<=(const iterator& it) const
{
    return ptr <= it.ptr;
}

bool paz::MatRef::iterator::operator>=(const iterator& it) const
{
    return ptr >= it.ptr;
}

bool paz::MatRef::iterator::operator<(const iterator& it) const
{
    return ptr < it.ptr;
}

bool paz::MatRef::iterator::operator>(const iterator& it) const
{
    return ptr > it.ptr;
}

paz::MatRef::iterator paz::operator+(MatRef::iterator::difference_type n, const
    MatRef::iterator& it)
{
    return it + n;
}

paz::MatRef::MatRef(const Mat& m, std::size_t startRow, std::size_t startCol,
    std::size_t numRows, std::size_t numCols) : _begin({m.data() + startRow +
    m.rows()*startCol, 0, static_cast<iterator::difference_type>(m.rows()),
    static_cast<iterator::difference_type>(numRows)}), _origCols(m.cols()),
    _blockCols(numCols) {}

paz::MatRef::MatRef(const Mat& m) : MatRef(m, 0, 0, m.rows(), m.cols()) {}

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

double paz::MatRef::operator()(std::size_t i, std::size_t j) const
{
    if(i >= rows())
    {
        throw std::runtime_error("Row index out of range.");
    }
    if(j >= cols())
    {
        throw std::runtime_error("Column index out of range.");
    }
    return _begin[i + rows()*j];
}

double paz::MatRef::operator()(std::size_t i) const
{
    if(i >= size())
    {
        throw std::runtime_error("Index is out of range.");
    }
    return _begin[i];
}

std::size_t paz::MatRef::size() const
{
    return rows()*cols();
}

std::size_t paz::MatRef::rows() const
{
    return _begin.blockRows;
}

std::size_t paz::MatRef::cols() const
{
    return _blockCols;
}

bool paz::MatRef::empty() const
{
    return !rows() || !cols();
}

const paz::MatRef::iterator& paz::MatRef::begin() const
{
    return _begin;
}

paz::MatRef::iterator paz::MatRef::end() const
{
    return _begin + size();
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

paz::MatRef paz::MatRef::block(std::size_t startRow, std::size_t startCol, std::
    size_t numRows, std::size_t numCols) const
{
throw std::logic_error("RETURNING TEMP"); //TEMP
    return Mat(*this).block(startRow, startCol, numRows, numCols);
}

paz::MatRef paz::MatRef::row(std::size_t m) const
{
throw std::logic_error("RETURNING TEMP"); //TEMP
    return Mat(*this).row(m);
}

paz::MatRef paz::MatRef::col(std::size_t n) const
{
throw std::logic_error("RETURNING TEMP"); //TEMP
    return Mat(*this).col(n);
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
