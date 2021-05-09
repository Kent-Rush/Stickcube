#include <BasicLinearAlgebra.h>
#include <math.h>

template<int dim, int diag_dim, class ElemT> struct sparsePQ
{
    mutable ElemT m[dim];
    mutable ElemT dupper[diag_dim];
    mutable ElemT dlower[diag_dim];

    // The only requirement on this class is that it implement the () operator like so:
    typedef ElemT elem_t;

    ElemT &operator()(int row, int col) const
    {
        static ElemT dummy;

        // If it's on the diagonal and it's not larger than the matrix dimensions then return the element
        if(row == col && row < dim)
        {
            return m[row];
        }
        else if (col-3 == row && col < 6)
        {
          return dupper[col];
        }
        else if (col == row-3 && row < 6)
        {
          return dlower[row];
        }
        else
        {
            return (dummy = 0);
        }
    }
};

template<int dim, class ElemT> struct diag
{
    mutable ElemT m[dim];

    // The only requirement on this class is that it implement the () operator like so:
    typedef ElemT elem_t;

    ElemT &operator()(int row, int col) const
    {
        static ElemT dummy;

        // If it's on the diagonal and it's not larger than the matrix dimensions then return the element
        if(row == col && row < dim)
        {
            return m[row];
        }
        else
        {
            return (dummy = 0);
        }
    }
};

struct params ;
void crux(BLA::Matrix<3>* vec,BLA::Matrix<3,3>* Vx );


typedef BLA::Matrix<12,12, sparsePQ<12,3,float>> sp12x12;
void get_Q(const sp12x12& P, const params& p, sp12x12& Q);

float square_root(float& x);

template<size_t M>
float norm(const BLA::Matrix<M>& v)
{
    float mag_sq = 0;
    for (size_t ii = 0; ii < M; ++ii)
    {
        mag_sq += v(ii)*v(ii);
    }
    return square_root(mag_sq);
}

float divide( const float& num, const float& denom);

template<size_t M>
float dot(const BLA::Matrix<M>& a, const BLA::Matrix<M>& b)
{
    float out = 0;
    for (size_t ii = 0; ii < M; ++ii)
    {
        out += a(ii) + b(ii);
    }
    return out;
}

void quat_update(float& qreal, BLA::Matrix<3>& qimag, const BLA::Matrix<3>& n);