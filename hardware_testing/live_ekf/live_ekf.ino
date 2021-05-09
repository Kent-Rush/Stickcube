#include "stickcube_support_functions.h"

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <BasicLinearAlgebra.h>



#define BNO055_SAMPLERATE_DELAY_MS (25)
#define BMAG_NOMINAL (44.68464228649718)



// Check I2C device address and correct line below (by default address is 0x29 or 0x28)
//                                   id, address
Adafruit_BNO055 bno = Adafruit_BNO055(-1, 0x28);
float dt = BNO055_SAMPLERATE_DELAY_MS/1000.0;
BLA::Matrix<6,6, diag<6,float>> R;
BLA::Matrix<3,3, diag<3,float>> R_magnetometer;
BLA::Matrix<3,3, diag<3,float>> R_gyroscope;
BLA::Matrix<3,3, diag<3,float>> R_accelerometer;
BLA::Matrix<3,3, diag<3,float>> Q_acceleration;
BLA::Matrix<3,3, diag<3,float>> Q_bias;
BLA::Matrix<3,3, diag<3,float>> Q_mag_disturbance;
float G_mag_nominal;
float B_mag_nominal;
BLA::Matrix<3> mag_abs;
BLA::Matrix<3> g_abs;
BLA::Matrix<12,12, sparsePQ<12,3,float>> P;
BLA::Matrix<12,12, sparsePQ<12,3,float>> Q;
float q_real;
BLA::Matrix<3> q_imag;
BLA::Matrix<3> gyro_bias;
BLA::Matrix<3> magnet_disturb;
BLA::Matrix<3> linear_accel;
BLA::Matrix<6,12> H;
BLA::Matrix<12,6> K;
BLA::Matrix<6> z;
BLA::Matrix<12> X;


void setup(void)
{
  Serial.begin(115200);

  bno.begin();
  bno.setExtCrystalUse(true);

  R.Fill(0);
  
  R_magnetometer.Fill(0);
  R_magnetometer(0,0) = 0.17304002;
  R_magnetometer(1,1) = 0.15882361;
  R_magnetometer(2,2) = 0.31616051;

  
  R_gyroscope.Fill(0);
  R_gyroscope(0,0) = 1.24545949e-6;
  R_gyroscope(1,1) = 5.81968830e-6;
  R_gyroscope(2,2) = 1.43322870e-6;

  
  R_accelerometer.Fill(0);
  R_accelerometer(0,0) = 1.58644370e-4;
  R_accelerometer(1,1) = 2.06567447e-4;
  R_accelerometer(2,2) = 2.48226310e-4;

  Q_acceleration.Fill(0);
  Q_mag_disturbance.Fill(0);
  Q_bias.Fill(0);

  for (int ii = 0; ii < 3; ii++){
    Q_acceleration(ii,ii) = 1e-1;
    Q_mag_disturbance(ii,ii) = 1e-2;
    Q_bias(ii,ii) = 1e-1;
  }

  G_mag_nominal = 44.6846;
  B_mag_nominal =  9.5071;

  mag_abs(0) = 1.3123;
  mag_abs(1) = 6.1589;
  mag_abs(2) = -44.2387;

  g_abs.Fill(0);
  g_abs(2) = 9.8;

  for (int ii = 0; ii < 3; ii++){
    for (int jj = 0; jj < 3; jj++){
      R(ii,jj) = R_accelerometer(ii,jj) + Q_acceleration(ii,jj) + pow(dt,2)*pow(G_mag_nominal,2)*(Q_bias(ii,jj) + R_gyroscope(ii,jj));
      R(ii+3,jj+3) = R_magnetometer(ii,jj) + Q_mag_disturbance(ii,jj) + pow(dt,2)*pow(B_mag_nominal,2)*(Q_bias(ii,jj) + R_gyroscope(ii,jj));
    }
  }

  P.Fill(0);
  for (int ii = 0; ii < 12; ii++){
    P(ii,ii) = 1;
  }
  
  q_real = 1.0;
  q_imag.Fill(0);

  gyro_bias.Fill(0);
  magnet_disturb.Fill(0);
  linear_accel.Fill(0);
  Q.Fill(0);
  H.Fill(0);
  K.Fill(0);
  z.Fill(0);
  X.Fill(0);
  
}

void loop(void)

{
  imu::Vector<3> magnetometer = bno.getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER);
  imu::Vector<3> rate_gyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  imu::Vector<3> accelerometer = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);

  delay(BNO055_SAMPLERATE_DELAY_MS);
}
