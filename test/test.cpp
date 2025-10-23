#include "PAZ_Math"
#include <iostream>
#include <iomanip>
#include <sstream>

#define PRINT(x) std::cout << #x << std::endl << x << std::endl << std::endl;
#define PRINT2(x) out << #x << std::endl << x << std::endl << std::endl;

int main()
{
    paz::Mat temp0(7, 11);
    const auto temp1 = temp0.block(0, 0, 3, 5);
    auto it = temp1.end();
    PRINT(std::distance(temp1.begin(), it - 10))
    PRINT(std::distance(temp1.begin(), it - 6))
    PRINT(std::distance(temp1.begin(), it + 0))
    PRINT(std::distance(temp1.begin(), it + 6))
    PRINT(std::distance(temp1.begin(), it + 10))

    auto l = paz::Mat::Zero(2);
    l(0, 0) = paz::randn();
    l(1, 0) = paz::randn();
    l(1, 1) = paz::randn();
    const auto a = l*l.trans();
    const paz::Mat m = 0.01*paz::Mat::Randn(2, 3);
    const auto c = paz::randn();

    PRINT(a)
    PRINT(a.det())

    PRINT(a.inv())
    PRINT(a.inv().det())

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

    PRINT(m.row(0).trans().cross(m.row(1).trans()))
    PRINT(paz::Mat::Cross(m.row(0).trans())*m.row(1).trans())
    {
        const paz::Mat x = paz::Mat::Cross(m.row(0).trans())*m.row(1).trans();
        std::ostringstream out;
        out << std::fixed;
        out << x << std::endl << std::endl;
        out << std::setprecision(4);
        out << x << std::endl << std::endl;
        out << std::setw(10);
        out << x << std::endl << std::endl;
        std::cout << out.str();
    }

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

    m34.shuffleCols();
    PRINT(m34)

    paz::Mat seq(7, 11);
    std::iota(seq.begin(), seq.end(), std::size_t{0});
    {
        std::ostringstream out;
        out << std::fixed << std::setprecision(0);
        PRINT2(seq)
        PRINT2(seq.block(1, 2, 3, 5));
        PRINT2(seq.block(1, 2, 3, 5).block(1, 1, 2, 4))
        std::cout << out.str();
    }

    {
        std::vector<std::size_t> rowSols;
        PRINT(jv(paz::Mat{{1.}}, rowSols))
        PRINT(jv(paz::Mat{{1., 2., 3.}}, rowSols))
        PRINT(jv(paz::Mat{{0., paz::inf()}}, rowSols))
        PRINT(jv(paz::Mat{{paz::inf(), 0.}}, rowSols))
    }

    {
        paz::Vec p = paz::Vec::Randn(4).normalized();
        paz::Vec q = paz::Vec::Randn(4).normalized();
        PRINT(qmult(p, q).trans())
    }

    {
        std::vector<paz::complex> v(8);
        for(auto& n : v)
        {
            n = {paz::randn(), paz::randn()};
        }
        const std::vector<paz::complex> u = paz::ifft(paz::fft(v));
        std::cout << "v                     ifft(fft(v))" << std::endl;
        std::ostringstream out;
        out << std::fixed;
        for(std::size_t i = 0; i < v.size(); ++i)
        {   out << std::setw(21) << v[i] << ' ' << std::setw(21) << u[i] << std::endl;
        }
        std::cout << out.str() << std::endl;
    }
}
