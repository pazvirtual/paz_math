#include "PAZ_Math"

int main()
{
    paz::Mat a(2);
    a(0, 0) = paz::randn();
    a(1, 0) = paz::randn();
    a(1, 1) = paz::randn();
    const auto b = a*a.trans();
    const auto c = b.chol();
    const auto d = c*c.trans();
    std::cout << a << std::endl << std::endl;
    std::cout << b << std::endl << std::endl;
    std::cout << c << std::endl << std::endl;
    std::cout << d << std::endl << std::endl;
}
