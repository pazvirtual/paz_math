#include "PAZ_Math"

paz::BlockRef::BlockRef(Mat& mat, std::size_t startRow, std::size_t startCol,
    std::size_t numRows, std::size_t numCols) : _baseData(mat.data()),
    _baseRows(mat.rows()), _startRow(startRow), _startCol(startCol), _rows(
    numRows), _cols(numCols) {}

paz::BlockRef::operator Mat() const
{
    paz::Mat res(_rows, _cols);
    for(std::size_t i = 0; i < _rows; ++i)
    {
        for(std::size_t j = 0; j < _cols; ++j)
        {
            res(i, j) = *(_baseData + _startRow + i + _baseRows*(_startCol +
                j));
        }
    }
    return res;
}

paz::BlockRef& paz::BlockRef::operator=(const paz::Mat& rhs)
{
    if(_rows != rhs.rows() || _cols != rhs.cols())
    {
        throw std::runtime_error("Matrix dimensions do not match.");
    }
    for(std::size_t i = 0; i < _rows; ++i)
    {
        for(std::size_t j = 0; j < _cols; ++j)
        {
            *(_baseData + i + _baseRows*j) = rhs(i, j);
        }
    }
    return *this;
}
