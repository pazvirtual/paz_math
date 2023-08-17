#include "PAZ_Math"

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

    PRINT(a.chol())
    PRINT(m)
    PRINT(c)
    PRINT(a.chol().cholUpdate(m, c))
    PRINT((a + m*m.trans()*c).chol())

    auto aNew = a;
    aNew.setCol(0, paz::Vec::Ones(aNew.rows()));
    PRINT(aNew)
    PRINT(aNew.eig())

    paz::Vec z(aNew.rows()*aNew.cols());
    std::copy(aNew.begin(), aNew.end(), z.begin());
    PRINT(z);
    z.setHead(2, z.tail(2));
    PRINT(z)
}
