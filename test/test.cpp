#include "PAZ_Math"
#include <iostream>

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
}
