#include <BasicLinearAlgebra.h>

void crux(BLA::Matrix<3>* vec,BLA::Matrix<3,3>* Vx )
{
  Vx->Fill(0);
  (*Vx)(0,1) = -(*vec)(2);
  (*Vx)(0,2) =  (*vec)(1);
  (*Vx)(1,0) =  (*vec)(2);
  (*Vx)(1,2) = -(*vec)(0);
  (*Vx)(2,0) = -(*vec)(1);
  (*Vx)(2,1) =  (*vec)(0);
}
