#include "PAZ_Math"

#define PRINT(x) std::cout << #x << std::endl << x << std::endl << std::endl;

int main()
{
    paz::Mat l(2);
    l(0, 0) = paz::randn();
    l(1, 0) = paz::randn();
    l(1, 1) = paz::randn();
    const paz::Mat a = l*l.trans();
    paz::Mat m(2, 3);
    for(auto& n : m)
    {
        n = 0.01*paz::randn();
    }
    const double c = paz::randn();

    PRINT(a)
    PRINT(a.chol())
    PRINT(m)
    PRINT(c)
    PRINT(a.chol().cholUpdate(m, c))
    PRINT((a + m*m.trans()*c).chol())
}
