#include "PAZ_Math"

int main()
{
    paz::Mat a(4);
    for(auto& n : a)
    {
        n = paz::randn();
    }
    paz::Vec v(4);
    for(auto& n : v)
    {
        n = paz::randn();
    }
    std::cout << a << std::endl << std::endl;
    std::cout << v << std::endl << std::endl;
    v.head(2) = paz::Vec::Ones(2);
    a.col(0) = v;
    std::cout << a << std::endl << std::endl;
}
