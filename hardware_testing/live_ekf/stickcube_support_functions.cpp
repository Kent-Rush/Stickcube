#include <BasicLinearAlgebra.h>
#include "stickcube_support_functions.h"
#include <math.h>
#include <limits.h>



struct params 
{
    BLA::Matrix<3,3, diag<3,float>> R_magnetometer;
    BLA::Matrix<3,3, diag<3,float>> R_gyroscope;
    BLA::Matrix<3,3, diag<3,float>> R_accelerometer;
    BLA::Matrix<3,3, diag<3,float>> Q_acceleration;
    BLA::Matrix<3,3, diag<3,float>> Q_bias;
    BLA::Matrix<3,3, diag<3,float>> Q_mag_disturbance;
    float ca;
    float cd;
    float dt;
};

void crux(const BLA::Matrix<3>& vec,BLA::Matrix<3,3>& Vx )
{
  Vx(0,1) = -vec(2);
  Vx(0,2) =  vec(1);
  Vx(1,0) =  vec(2);
  Vx(1,2) = -vec(0);
  Vx(2,0) = -vec(1);
  Vx(2,1) =  vec(0);
  for (size_t ii = 0; ii < 3; ++ii)
  {
    Vx(ii,ii) = 0.0;
  }
}

void get_Q(const sp12x12& P, const params& p, sp12x12& Q)
{
    for (size_t ii = 0; ii < 3; ++ii)
    {
        Q(ii,ii) = P(ii,ii) + p.dt*p.dt*(P(ii,ii) + p.Q_bias(ii,ii) + p.R_gyroscope(ii,ii));
        Q(ii,ii+3)  = -p.dt*(P(ii,ii+3) + p.Q_bias(ii,ii));
        Q(ii+3,ii) = Q(ii,ii+3);
        Q(ii+3,ii+3) = P(ii+3,ii+3) + p.Q_bias(ii,ii);
        Q(ii+6,ii+6) = p.ca*p.ca*P(ii+6,ii+6) + p.Q_acceleration(ii,ii);
        Q(ii+9,ii+9) = p.cd*p.cd*P(ii+9,ii+9) + p.Q_mag_disturbance(ii,ii);
    }
}

void getH(const BLA::Matrix<3>& g_gyro, const BLA::Matrix<3>& m_gyro, const params& p, BLA::Matrix<6,12>& H)
{
    BLA::Matrix<3,3> g_gyro_cross;
    crux(g_gyro, g_gyro_cross);
    BLA::Matrix<3,3> m_gyro_cross;
    crux(m_gyro, m_gyro_cross);
    for (size_t ii = 0; ii < 3; ++ii)
    {
        for (size_t jj = 0; jj < 3; ++jj)
        {
            H(ii,jj) = -g_gyro_cross(ii,jj);
            H(ii,jj+3) = p.dt*g_gyro_cross(ii,jj);
            H(ii+3,jj) = -m_gyro_cross(ii,jj);
            H(ii+3,jj+3) = p.dt*m_gyro_cross(ii,jj);
            if (ii == jj)
            {
                H(ii,jj+6) = 0.0;
                H(ii+3,jj+9) = 0.0;
            }
            else
            {
                H(ii, jj+6)     = 0.0;
                H(ii+3, jj+6)   = 0.0;
                H(ii,jj+9)      = 0.0;
                H(ii+3, jj+9)   = 0.0;
            }
        }
    }
}

float square_root(float& x)
{
    if (x < 0)
    {
        return 0;
    }
    else
    {
        return sqrt(x);
    }
}



float divide( const float& num, const float& denom)
{
    if (denom < 1e-12)
    {
        return 0;
    }
    else
    {
        return num/denom;
    }
}




void quat_update(float& qreal, BLA::Matrix<3>& qimag, const BLA::Matrix<3>& n)
{
    float angle = norm(n)
    BLA::Matrix<3> axis;
    for (size_t ii = 0; ii < 3; ++ii)
    {
        axis(ii) = divide(n(ii),angle);
    }

    float real;
    BLA::Matrix<3> imag;

    imag = imag*sin(real/2);
    real = cos(real/2);

    BLA::Matrix<3,3> imag_crux;
    crux(imag,imag_crux);
    float imag_dot = dot(imag, qimag);

    qimag = real*qimag + qreal*imag + imag_crux*qimag;
    qreal = real*qreal - imag_dot;


}