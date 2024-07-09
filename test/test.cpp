#include "PAZ_Math"
#include <iostream>
#include <iomanip>

#define PRINT(x) std::cout << #x << std::endl << x << std::endl << std::endl;

int main()
{
    auto l = paz::Mat::Zero(2);
    l(0, 0) = paz::randn();
    l(1, 0) = paz::randn();
    l(1, 1) = paz::randn();
    const auto a = l*l.trans();
    paz::Mat m(2, 3);
    for(auto& n : m)
    {
        n = 0.01*paz::randn();
    }
    const auto c = paz::randn();

    PRINT(a)
    PRINT(a.det())
    PRINT(std::log10(std::abs(a.det() - 1./a.inv().det())))

    PRINT(a.diag())
    PRINT(paz::Mat::Diag(a.diag()))

    paz::Mat vecs;
    const paz::Mat vals = a.eig(vecs);
    PRINT(vals);
    PRINT(vecs);

    PRINT(m.trans())
    paz::Mat q;
    paz::Mat r;
    std::vector<std::size_t> p;
    m.trans().qr(q, r, p);
    PRINT(q*r)
    std::cout << 'p' << std::endl;
    for(std::size_t i = 0; i < p.size(); ++i)
    {
        std::cout << p[i] << (i + 1 < p.size() ? " " : "");
    }
    std::cout << std::endl << std::endl;
    PRINT(m.col(0).rep(2, 3))

    try
    {
        PRINT(a.chol())
        PRINT(m)
        PRINT(c)
        PRINT(a.chol().cholUpdate(m, c))
        PRINT((a + m*m.trans()*c).chol())
    }
    catch(const std::exception& e)
    {
        std::cerr << "Warning: " << e.what() << std::endl; //TEMP - remove randomness and add proper unit tests
    }

    auto aNew = a;
    aNew.setCol(0, paz::Vec::Ones(aNew.rows()));
    PRINT(aNew)
    PRINT(aNew.eig())

    paz::Vec z(aNew.rows()*aNew.cols());
    std::copy(aNew.begin(), aNew.end(), z.begin());
    PRINT(z);
    z.setHead(2, z.tail(2));
    PRINT(z)

    PRINT(static_cast<paz::Vec>(m.row(0).trans()).cross(m.row(1).trans()))
    PRINT(paz::Mat::Cross(m.row(0).trans())*m.row(1).trans())

    const paz::Mat x = paz::Mat::Cross(m.row(0).trans())*m.row(1).trans();
    std::cout << std::fixed;
    std::cout << x << std::endl << std::endl;
    std::cout << std::setprecision(4);
    std::cout << x << std::endl << std::endl;
    std::cout << std::setw(10);
    std::cout << x << std::endl << std::endl;

    PRINT(paz::mix(a, aNew, 0.1))

    paz::Mat m33 = paz::Mat(3, 3);
    for(auto& n : m33)
    {
        n = paz::randn();
    }
    PRINT(m33)
    auto m34 = m33;
    m34.resizeCols(4);
    PRINT(m34)
    auto m43 = m33;
    m43.resizeRows(4);
    PRINT(m43)
    auto m44 = m33;
    m44.resize(4, 4);
    PRINT(m44)
    auto m32 = m33;
    m32.resizeCols(2);
    PRINT(m32)
    auto m23 = m33;
    m23.resizeRows(2);
    PRINT(m23)
    auto m22 = m33;
    m22.resize(2, 2);
    PRINT(m22)

    const paz::Vec v3 = m33.col(0);
    PRINT(v3);
    auto v4 = v3;
    v4.resize(4);
    PRINT(v4);
    auto v2 = v3;
    v2.resize(2);
    PRINT(v2);

    PRINT(paz::Mat::Hcat(m34, m32))
    PRINT(paz::Mat::Vcat(m43, m23))
}
