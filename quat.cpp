#include "PAZ_Math"

paz::Mat paz::to_mat(const Vec& q)
{
    if(q.size() != 4)
    {
        throw std::runtime_error("Not a quaternion.");
    }
    const auto xx = q(0)*q(0);
    const auto yy = q(1)*q(1);
    const auto zz = q(2)*q(2);
    const auto xy = q(0)*q(1);
    const auto zw = q(2)*q(3);
    const auto xz = q(0)*q(2);
    const auto yw = q(1)*q(3);
    const auto yz = q(1)*q(2);
    const auto xw = q(0)*q(3);
    return {{1. - 2.*(yy + zz),      2.*(xy + zw),      2.*(xz - yw)},
            {     2.*(xy - zw), 1. - 2.*(xx + zz),      2.*(yz + xw)},
            {     2.*(xz + yw),      2.*(yz - xw), 1. - 2.*(xx + yy)}};
}

paz::Vec paz::to_quat(const Mat& m)
{
    if(m.rows() != 3 || m.cols() != 3)
    {
        throw std::runtime_error("Not a rotation matrix.");
    }
    const auto fourXSquaredMinus1 = m(0, 0) - m(1, 1) - m(2, 2);
    const auto fourYSquaredMinus1 = m(1, 1) - m(0, 0) - m(2, 2);
    const auto fourZSquaredMinus1 = m(2, 2) - m(0, 0) - m(1, 1);
    const auto fourWSquaredMinus1 = m(0, 0) + m(1, 1) + m(2, 2);
    int biggestIndex = 0;
    auto fourBiggestSquaredMinus1 = fourWSquaredMinus1;
    if(fourXSquaredMinus1 > fourBiggestSquaredMinus1)
    {
        fourBiggestSquaredMinus1 = fourXSquaredMinus1;
        biggestIndex = 1;
    }
    if(fourYSquaredMinus1 > fourBiggestSquaredMinus1)
    {
        fourBiggestSquaredMinus1 = fourYSquaredMinus1;
        biggestIndex = 2;
    }
    if(fourZSquaredMinus1 > fourBiggestSquaredMinus1)
    {
        fourBiggestSquaredMinus1 = fourZSquaredMinus1;
        biggestIndex = 3;
    }
    const auto biggestVal = sqrt(fourBiggestSquaredMinus1 + 1.)*0.5;
    const auto mult = 0.25/biggestVal;
    switch(biggestIndex)
    {
        case 0:
            return {{(m(1, 2) - m(2, 1))*mult,
                     (m(2, 0) - m(0, 2))*mult,
                     (m(0, 1) - m(1, 0))*mult,
                                   biggestVal}};
        case 1:
            return {{              biggestVal,
                     (m(0, 1) + m(1, 0))*mult,
                     (m(2, 0) + m(0, 2))*mult,
                     (m(1, 2) - m(2, 1))*mult}};
        case 2:
            return {{(m(0, 1) + m(1, 0))*mult,
                                   biggestVal,
                     (m(1, 2) + m(2, 1))*mult,
                     (m(2, 0) - m(0, 2))*mult}};
        default:
            return {{(m(2, 0) + m(0, 2))*mult,
                     (m(1, 2) + m(2, 1))*mult,
                                   biggestVal,
                     (m(0, 1) - m(1, 0))*mult}};
    }
}

paz::Vec paz::qinv(const Vec& q)
{
    if(q.size() != 4)
    {
        throw std::runtime_error("Not a quaternion.");
    }
    return {{-q(0), -q(1), -q(2), q(3)}};
}

paz::Mat paz::xi(const Vec& q)
{
    if(q.size() != 4)
    {
        throw std::runtime_error("Not a quaternion.");
    }
    return {{ q(3), -q(2),  q(1)},
            { q(2),  q(3), -q(0)},
            {-q(1),  q(0),  q(3)},
            {-q(0), -q(1), -q(2)}};
}

paz::Vec paz::qmult(const Vec& p, const Vec& q)
{
    const Vec pVec = p.head(3);
    const Vec qVec = q.head(3);
    const Vec rVec = q(3)*pVec + p(3)*qVec - pVec.cross(qVec);
    return {{rVec(0), rVec(1), rVec(2), p(3)*q(3) - pVec.dot(qVec)}};
}

paz::Vec paz::axis_angle(const Vec& axis, double angle)
{
    const double s = std::sin(0.5*angle);
    return Vec{{s*axis(0), s*axis(1), s*axis(2), std::cos(0.5*angle)}};
}
