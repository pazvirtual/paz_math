#include "PAZ_Math"

void paz::murty(const MatRef& costMat, std::size_t numBest, std::vector<std::
    vector<std::size_t>>& rowSols, std::vector<double>& costs)
{
    rowSols.clear();
    costs.clear();

    if(!numBest)
    {
        return;
    }

    std::vector<std::size_t> bestSol;
    const double bestCost = jv(costMat, bestSol);

    if(numBest == 1)
    {
        rowSols = {bestSol};
        costs = {bestCost};
        return;
    }

    const std::size_t rows = costMat.rows();
    const std::size_t cols = costMat.cols();
    if(rows > cols)
    {
        throw std::runtime_error("Matrix is too tall.");
    }

    std::vector<Mat> costMatsList = {costMat};
    std::vector<std::vector<std::size_t>> solsList = {bestSol};
    std::vector<double> costsList = {bestCost};

    while(true)
    {
        const std::size_t idx = std::distance(costsList.begin(), std::
            min_element(costsList.begin(), costsList.end()));
        costs.push_back(costsList[idx]);
        rowSols.push_back(solsList[idx]);

        if(rowSols.size() == numBest)
        {
            break;
        }

        Mat curCostMat = costMatsList[idx];
        const auto curSol = solsList[idx];

        std::swap(costMatsList[idx], costMatsList.back());
        costMatsList.pop_back();
        std::swap(solsList[idx], solsList.back());
        solsList.pop_back();
        std::swap(costsList[idx], costsList.back());
        costsList.pop_back();

        for(std::size_t i = 0; i < curSol.size(); ++i)
        {
            if(curSol[i] != None)
            {
                if(curCostMat(i, curSol[i]) == inf())
                {
                    throw std::logic_error("Current solution should have been s"
                        "kipped due to infinite cost.");
                }

                Mat tempCostMat = curCostMat;
                tempCostMat(i, curSol[i]) = inf();

                std::vector<std::size_t> tempSol;
                const double tempCost = jv(tempCostMat, tempSol);
                if(tempCost < inf() && std::find(tempSol.begin(), tempSol.
                    end(), None) == tempSol.end())
                {
                    costMatsList.push_back(tempCostMat);
                    solsList.push_back(tempSol);
                    costsList.push_back(tempCost);
                }

                const double temp = curCostMat(i, curSol[i]);
                for(std::size_t j = 0; j < cols; ++j)
                {
                    curCostMat(i, j) = inf();
                }
                for(std::size_t j = 0; j < rows; ++j)
                {
                    curCostMat(j, curSol[i]) = inf();
                }
                curCostMat(i, curSol[i]) = temp;
            }
        }

        if(costsList.empty())
        {
            break;
        }
    }
}
