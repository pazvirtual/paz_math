#include "PAZ_Math"
#include <cstdint>

#define GET_COST(i, j) get_cost(costMat, resolution, intInf, i, j)

constexpr double MinRes = 1e-9;

// Handle non-square matrices. (1/2)
static std::int64_t get_cost(const paz::Mat& costMat, double resolution, std::
    int64_t intInf, std::size_t i, std::size_t j)
{
    if(i < costMat.rows())
    {
        const double c = costMat(i, j);
        if(std::isfinite(c))
        {
            return std::round(c/resolution);
        }
        return intInf;
    }
    return 0;
}

static double max_finite_cost(const paz::Mat& costMat)
{
    double maxCost = -paz::inf();
    for(auto val : costMat)
    {
        if(std::isfinite(val) && val > maxCost)
        {
            maxCost = val;
        }
    }
    return maxCost;
}

double paz::jv(const Mat& costMat, std::vector<std::size_t>& rowSols)
{
    // Find discretization parameters and prepare cost matrix.
    const std::size_t rows = costMat.rows();
    const std::size_t cols = costMat.cols();
    if(rows > cols)
    {
        throw std::runtime_error("Matrix is too tall.");
    }

    const double minCost = costMat.min();

    if(minCost < 0.)
    {
        throw std::runtime_error("All costs must be non-negative.");
    }
    if(!std::isfinite(minCost))
    {
        throw std::runtime_error("All costs are infinite.");
    }

    const double maxCost = max_finite_cost(costMat);

    const double resolution = std::max(MinRes, paz::eps(maxCost*(rows + 1)));
    const std::int64_t intInf = std::round(maxCost/resolution)*(rows + 1);

    rowSols.resize(cols, None);
    std::vector<std::size_t> colSols(cols, None);

    std::vector<std::size_t> colList(cols);
    std::iota(colList.begin(), colList.end(), std::size_t{0});

    std::vector<std::int64_t> colCosts(cols);
    std::vector<bool> skip(cols, false);
    std::int64_t h, u1, u2;
    std::size_t j, j1, f0, f;

    // 1. Column reduction.
    for(std::size_t idx = 0; idx < cols; ++idx)
    {
        j = cols - idx - 1;
        h = GET_COST(0, j);
        std::size_t i1 = 0;
        for(std::size_t i = 1; i < cols; ++i)
        {
            if(GET_COST(i, j) < h)
            {
                h = GET_COST(i, j);
                i1 = i;
            }
        }
        colCosts[j] = h;
        if(rowSols[i1] == None)
        {
            rowSols[i1] = j;
            colSols[j] = i1;
        }
        else
        {
            skip[i1] = true;
        }
    }

    // 2. Reduction transfer.
    std::vector<std::size_t> free(cols);
    f = 0;
    for(std::size_t i = 0; i < cols; ++i)
    {
        if(rowSols[i] == None)
        {
            free[f] = i;
            ++f;
        }
        else if(!skip[i])
        {
            j1 = rowSols[i];
            auto minTEMP = std::numeric_limits<std::int64_t>::max();
            for(std::size_t j = 0; j < cols; ++j)
            {
                if(j != j1)
                {
                    minTEMP = std::min(minTEMP, GET_COST(i, j) - colCosts[j]);
                }
            }
            colCosts[j1] -= minTEMP;
        }
    }

    // 3. Augmenting row reduction.
    for(int cnt = 0; cnt < 2; ++cnt)
    {
        std::size_t k = 0;
        f0 = f;
        f = 0;
        while(k < f0)
        {
            const std::size_t i = free[k];
            ++k;
            u1 = GET_COST(i, 0) - colCosts[0];
            j1 = 0;
            std::size_t j2 = 0;
            u2 = std::numeric_limits<std::int64_t>::max();
            for(j = 1; j < cols; ++j)
            {
                h = GET_COST(i, j) - colCosts[j];
                if(h < u2)
                {
                    if(h >= u1)
                    {
                        u2 = h;
                        j2 = j;
                    }
                    else
                    {
                        u2 = u1;
                        u1 = h;
                        j2 = j1;
                        j1 = j;
                    }
                }
            }
            std::size_t i1 = colSols[j1];
            if(u1 < u2)
            {
                colCosts[j1] += u1 - u2;
            }
            else if(i1 != None)
            {
                j1 = j2;
                i1 = colSols[j1];
            }
            if(i1 != None)
            {
                if(u1 < u2)
                {
                    --k;
                    free[k] = i1;
                }
                else
                {
                    free[f] = i1;
                    ++f;
                }
            }
            rowSols[i] = j1;
            colSols[j1] = i;
        }
    }

    // 4. Augmentation.
    std::vector<std::int64_t> pathLengths(cols);
    std::vector<std::size_t> pred(cols);
    f0 = f;
    for(f = 0; f < f0; ++f)
    {
        auto minTEMP = std::numeric_limits<std::int64_t>::max();
        std::size_t i = 0;
        std::size_t i1 = free[f];
        std::size_t low = 0;
        std::size_t up = 0;
        std::size_t last = None;
        for(j = 0; j < cols; ++j)
        {
            pathLengths[j] = GET_COST(i1, j) - colCosts[j];
            pred[j] = i1;
        }
        while(true)
        {
            if(up == low)
            {
                last = low ? low - 1 : None;
                minTEMP = pathLengths[colList[up]];
                ++up;
                for(std::size_t k = up; k < cols; ++k)
                {
                    j = colList[k];
                    h = pathLengths[j];
                    if(h <= minTEMP)
                    {
                        if(h < minTEMP)
                        {
                            up = low;
                            minTEMP = h;
                        }
                        colList[k] = colList[up];
                        colList[up] = j;
                        ++up;
                    }
                }
                for(std::size_t hTEMP = low; hTEMP < up; ++hTEMP)
                {
                    j = colList[hTEMP];
                    if(colSols[j] == None)
                    {
                        goto augment;
                    }
                }
            }

            j1 = colList[low];
            ++low;
            i = colSols[j1];
            u1 = GET_COST(i, j1) - colCosts[j1] - minTEMP;
            for(std::size_t k = up; k < cols; ++k)
            {
                j = colList[k];
                h = GET_COST(i, j) - colCosts[j] - u1;
                if(h < pathLengths[j])
                {
                    pathLengths[j] = h;
                    pred[j] = i;
                    if(h == minTEMP)
                    {
                        if(colSols[j] == None)
                        {
                            goto augment;
                        }
                        else
                        {
                            colList[k] = colList[up];
                            colList[up] = j;
                            ++up;
                        }
                    }
                }
            }
        }

    augment:
        if(last != None) //TEMP - not sure
        {
            for(std::size_t k = 0; k < last; ++k)
            {
                j1 = colList[k];
                colCosts[j1] = colCosts[j1] + pathLengths[j1] - minTEMP;
            }
        }
        std::size_t k;
        while(i != i1)
        {
            i = pred[j];
            colSols[j] = i;
            k = j;
            j = rowSols[i];
            rowSols[i] = k;
        }
    }

    // Handle non-square matrices. (2/2)
    rowSols.resize(rows);
    rowSols.shrink_to_fit();
    std::int64_t cost = 0;
    for(std::size_t i = 0; i < rows; ++i)
    {
        cost += GET_COST(i, rowSols[i]);
    }
    if(cost >= intInf)
    {
        return paz::inf();
    }
    return cost*resolution;
}
