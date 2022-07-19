/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "initial_alignment.h"
#include<unsupported/Eigen/Polynomials>

void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    delta_bg = A.ldlt().solve(b);
    std::cout << "gyroscope bias initial calibration " << delta_bg.transpose() << std::endl;

    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}


MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;


            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(n_state - 3);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
    }   
    g = g0;
}

void RefineGravityStereo(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2; // + 1

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 8);//9
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;


            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            // tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0]
                                     - frame_i->second.R.transpose() * dt * dt / 2 * g0
                                     - frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T);

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<2, 2>() += r_A.bottomRightCorner<2, 2>();
            b.tail<2>() += r_b.tail<2>();

            A.block<6, 2>(i * 3, n_state - 2) += r_A.topRightCorner<6, 2>();
            A.block<2, 6>(n_state - 2, i * 3) += r_A.bottomLeftCorner<2, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(n_state - 2);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
    }   
    g = g0;
}

bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;
    std::cout << "estimated scale: " <<  s << std::endl;
    g = x.segment<3>(n_state - 4);
    std::cout << " result g     " << g.norm() << " " << g.transpose() << std::endl;
    if(fabs(g.norm() - G.norm()) > 0.5 || s < 0)
    {
        return false;
    }

    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    std::cout << " refine     " << g.norm() << " " << g.transpose() << std::endl;
    if(s < 0.0 )
        return false;   
    else
        return true;
}

// scale=1.0, fixed
bool LinearAlignmentStereo(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 9);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        // tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0]
                                - frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T);
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
        b.tail<3>() += r_b.tail<3>();

        A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
        A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b); //x(9,1)
    // double s = x(n_state - 1) / 100.0;
    // std::cout << "estimated scale: " <<  s << std::endl;
    g = x.segment<3>(n_state - 3);
    std::cout << " result g     " << g.norm() << " " << g.transpose() << std::endl;
    if(fabs(g.norm() - G.norm()) > 0.5)// || s < 0
    {
        return false;
    }

    RefineGravityStereo(all_image_frame, g, x);
    // s = (x.tail<1>())(0) / 100.0;
    // (x.tail<1>())(0) = s;
    std::cout << " refine     " << g.norm() << " " << g.transpose() << std::endl;
    // if(s < 0.0 )
    //     return false;   
    // else
        return true;
}

bool FindLagrange(const Matrix3d& D, const Vector3d& d, double& lambda){
    double g = 9.81;
    double m00, m10, m20, m11, m21, m22;
    m00 = D(0, 0);
    m10 = D(1, 0);
    m20 = D(2, 0);
    m11 = D(1, 1);
    m21 = D(2, 1);
    m22 = D(2, 2);
    double d0, d1, d2;
    d0 = d(0);
    d1 = d(1);
    d2 = d(2);
    VectorXd coeff(7); // t^6, ... t^0
    coeff(0) = 1.0;
    coeff(1) = -1.0/(g*g)*((g*g)*m00*2.0+(g*g)*m11*2.0+(g*g)*m22*2.0);
    coeff(2) = -1.0/(g*g)*(d0*d0+d1*d1+d2*d2-(g*g)*(m00*m00)+(g*g)*(m10*m10)*2.0-(g*g)*(m11*m11)+(g*g)*(m20*m20)*2.0+(g*g)*(m21*m21)*2.0-(g*g)*(m22*m22)-(g*g)*m00*m11*4.0-(g*g)*m00*m22*4.0-(g*g)*m11*m22*4.0);
    coeff(3) = 1.0/(g*g)*((d1*d1)*m00*2.0+(d2*d2)*m00*2.0+(d0*d0)*m11*2.0+(d2*d2)*m11*2.0+(d0*d0)*m22*2.0+(d1*d1)*m22*2.0+(g*g)*m00*(m10*m10)*2.0-(g*g)*m00*(m11*m11)*2.0-(g*g)*(m00*m00)*m11*2.0+(g*g)*m00*(m20*m20)*2.0+(g*g)*m00*(m21*m21)*4.0+(g*g)*(m10*m10)*m11*2.0-(g*g)*m00*(m22*m22)*2.0-(g*g)*(m00*m00)*m22*2.0+(g*g)*m11*(m20*m20)*4.0+(g*g)*m11*(m21*m21)*2.0+(g*g)*(m10*m10)*m22*4.0-(g*g)*m11*(m22*m22)*2.0-(g*g)*(m11*m11)*m22*2.0+(g*g)*(m20*m20)*m22*2.0+(g*g)*(m21*m21)*m22*2.0-d0*d1*m10*4.0-d0*d2*m20*4.0-d1*d2*m21*4.0-(g*g)*m00*m11*m22*8.0-(g*g)*m10*m20*m21*4.0);
    coeff(4) = -1.0/(g*g)*((d1*d1)*(m00*m00)+(d2*d2)*(m00*m00)+(d0*d0)*(m10*m10)+(d0*d0)*(m11*m11)+(d1*d1)*(m10*m10)-(d2*d2)*(m10*m10)*2.0+(d2*d2)*(m11*m11)+(d0*d0)*(m20*m20)-(d0*d0)*(m21*m21)*2.0-(d1*d1)*(m20*m20)*2.0+(d0*d0)*(m22*m22)+(d1*d1)*(m21*m21)+(d2*d2)*(m20*m20)+(d1*d1)*(m22*m22)+(d2*d2)*(m21*m21)-(g*g)*(m10*m10*m10*m10)-(g*g)*(m20*m20*m20*m20)-(g*g)*(m21*m21*m21*m21)-(g*g)*(m00*m00)*(m11*m11)+(g*g)*(m00*m00)*(m21*m21)*2.0-(g*g)*(m00*m00)*(m22*m22)-(g*g)*(m10*m10)*(m20*m20)*2.0-(g*g)*(m10*m10)*(m21*m21)*2.0+(g*g)*(m11*m11)*(m20*m20)*2.0+(g*g)*(m10*m10)*(m22*m22)*2.0-(g*g)*(m11*m11)*(m22*m22)-(g*g)*(m20*m20)*(m21*m21)*2.0+(d2*d2)*m00*m11*4.0+(d1*d1)*m00*m22*4.0+(d0*d0)*m11*m22*4.0-d0*d1*m00*m10*2.0-d0*d1*m10*m11*2.0-d0*d2*m00*m20*2.0-d1*d2*m00*m21*8.0-d0*d1*m10*m22*8.0+d0*d2*m10*m21*6.0-d0*d2*m11*m20*8.0+d1*d2*m10*m20*6.0-d1*d2*m11*m21*2.0+d0*d1*m20*m21*6.0-d0*d2*m20*m22*2.0-d1*d2*m21*m22*2.0+(g*g)*m00*(m10*m10)*m11*2.0+(g*g)*m00*m11*(m20*m20)*4.0+(g*g)*m00*m11*(m21*m21)*4.0+(g*g)*m00*(m10*m10)*m22*4.0-(g*g)*m00*m11*(m22*m22)*4.0-(g*g)*m00*(m11*m11)*m22*4.0-(g*g)*(m00*m00)*m11*m22*4.0+(g*g)*m00*(m20*m20)*m22*2.0+(g*g)*m00*(m21*m21)*m22*4.0+(g*g)*(m10*m10)*m11*m22*4.0+(g*g)*m11*(m20*m20)*m22*4.0+(g*g)*m11*(m21*m21)*m22*2.0-(g*g)*m00*m10*m20*m21*4.0-(g*g)*m10*m11*m20*m21*4.0-(g*g)*m10*m20*m21*m22*4.0);
    coeff(5) = -1.0/(g*g)*((d2*d2)*m00*(m10*m10)*2.0-(d2*d2)*m00*(m11*m11)*2.0-(d2*d2)*(m00*m00)*m11*2.0+(d1*d1)*m00*(m20*m20)*2.0-(d1*d1)*m00*(m21*m21)*2.0-(d1*d1)*m00*(m22*m22)*2.0-(d1*d1)*(m00*m00)*m22*2.0-(d2*d2)*m00*(m21*m21)*2.0+(d2*d2)*(m10*m10)*m11*2.0-(d0*d0)*m11*(m20*m20)*2.0+(d0*d0)*m11*(m21*m21)*2.0-(d0*d0)*(m10*m10)*m22*2.0-(d0*d0)*m11*(m22*m22)*2.0-(d0*d0)*(m11*m11)*m22*2.0-(d1*d1)*(m10*m10)*m22*2.0-(d2*d2)*m11*(m20*m20)*2.0+(d0*d0)*(m21*m21)*m22*2.0+(d1*d1)*(m20*m20)*m22*2.0+(g*g)*m00*(m21*m21*m21*m21)*2.0+(g*g)*m11*(m20*m20*m20*m20)*2.0+(g*g)*(m10*m10*m10*m10)*m22*2.0+(g*g)*m00*(m10*m10)*(m21*m21)*2.0-(g*g)*m00*(m11*m11)*(m20*m20)*2.0-(g*g)*m00*(m10*m10)*(m22*m22)*2.0-(g*g)*(m00*m00)*m11*(m21*m21)*2.0+(g*g)*m00*(m11*m11)*(m22*m22)*2.0+(g*g)*(m00*m00)*m11*(m22*m22)*2.0+(g*g)*(m00*m00)*(m11*m11)*m22*2.0+(g*g)*m00*(m20*m20)*(m21*m21)*2.0+(g*g)*(m10*m10)*m11*(m20*m20)*2.0-(g*g)*(m00*m00)*(m21*m21)*m22*2.0-(g*g)*(m10*m10)*m11*(m22*m22)*2.0+(g*g)*m11*(m20*m20)*(m21*m21)*2.0+(g*g)*(m10*m10)*(m20*m20)*m22*2.0+(g*g)*(m10*m10)*(m21*m21)*m22*2.0-(g*g)*(m11*m11)*(m20*m20)*m22*2.0+d1*d2*(m00*m00)*m21*4.0+d0*d1*m10*(m22*m22)*4.0+d0*d2*(m11*m11)*m20*4.0+(d0*d0)*m10*m20*m21*4.0+(d1*d1)*m10*m20*m21*4.0+(d2*d2)*m10*m20*m21*4.0-(g*g)*m10*m20*(m21*m21*m21)*4.0-(g*g)*m10*(m20*m20*m20)*m21*4.0-(g*g)*(m10*m10*m10)*m20*m21*4.0-(g*g)*m00*(m10*m10)*m11*m22*4.0-(g*g)*m00*m11*(m20*m20)*m22*4.0-(g*g)*m00*m11*(m21*m21)*m22*4.0+d0*d1*m00*m10*m22*4.0-d0*d2*m00*m10*m21*4.0+d0*d2*m00*m11*m20*4.0-d1*d2*m00*m10*m20*4.0+d1*d2*m00*m11*m21*4.0-d0*d1*m00*m20*m21*4.0+d0*d1*m10*m11*m22*4.0-d0*d2*m10*m11*m21*4.0-d1*d2*m10*m11*m20*4.0+d1*d2*m00*m21*m22*4.0-d0*d1*m11*m20*m21*4.0-d0*d2*m10*m21*m22*4.0+d0*d2*m11*m20*m22*4.0-d1*d2*m10*m20*m22*4.0-d0*d1*m20*m21*m22*4.0+(g*g)*m00*m10*m11*m20*m21*4.0+(g*g)*m00*m10*m20*m21*m22*4.0+(g*g)*m10*m11*m20*m21*m22*4.0);
    coeff(6) = -1.0/(g*g)*((d2*d2)*(m10*m10*m10*m10)+(d0*d0)*(m21*m21*m21*m21)+(d1*d1)*(m20*m20*m20*m20)+(d2*d2)*(m00*m00)*(m11*m11)+(d1*d1)*(m00*m00)*(m21*m21)+(d1*d1)*(m00*m00)*(m22*m22)+(d2*d2)*(m00*m00)*(m21*m21)+(d0*d0)*(m10*m10)*(m21*m21)+(d0*d0)*(m11*m11)*(m20*m20)+(d1*d1)*(m10*m10)*(m20*m20)+(d0*d0)*(m10*m10)*(m22*m22)+(d2*d2)*(m10*m10)*(m20*m20)+(d0*d0)*(m11*m11)*(m22*m22)+(d1*d1)*(m10*m10)*(m22*m22)+(d2*d2)*(m10*m10)*(m21*m21)+(d2*d2)*(m11*m11)*(m20*m20)+(d0*d0)*(m20*m20)*(m21*m21)+(d1*d1)*(m20*m20)*(m21*m21)-(g*g)*(m00*m00)*(m21*m21*m21*m21)-(g*g)*(m11*m11)*(m20*m20*m20*m20)-(g*g)*(m10*m10*m10*m10)*(m22*m22)-d0*d2*m10*(m21*m21*m21)*2.0-d0*d2*(m10*m10*m10)*m21*2.0-d1*d2*m10*(m20*m20*m20)*2.0-d1*d2*(m10*m10*m10)*m20*2.0-d0*d1*m20*(m21*m21*m21)*2.0-d0*d1*(m20*m20*m20)*m21*2.0-(g*g)*(m00*m00)*(m11*m11)*(m22*m22)-(g*g)*(m10*m10)*(m20*m20)*(m21*m21)*4.0-(d2*d2)*m00*(m10*m10)*m11*2.0-(d1*d1)*m00*(m20*m20)*m22*2.0-(d0*d0)*m11*(m21*m21)*m22*2.0+(g*g)*m00*m10*m20*(m21*m21*m21)*4.0+(g*g)*m10*m11*(m20*m20*m20)*m21*4.0+(g*g)*(m10*m10*m10)*m20*m21*m22*4.0+(g*g)*m00*(m10*m10)*m11*(m22*m22)*2.0-(g*g)*m00*m11*(m20*m20)*(m21*m21)*2.0-(g*g)*m00*(m10*m10)*(m21*m21)*m22*2.0+(g*g)*m00*(m11*m11)*(m20*m20)*m22*2.0+(g*g)*(m00*m00)*m11*(m21*m21)*m22*2.0-(g*g)*(m10*m10)*m11*(m20*m20)*m22*2.0-d0*d1*m00*m10*(m21*m21)*2.0-d0*d1*m00*m10*(m22*m22)*2.0-d0*d2*m00*(m11*m11)*m20*2.0+d1*d2*m00*(m10*m10)*m21*2.0-d1*d2*(m00*m00)*m11*m21*2.0-d0*d1*m10*m11*(m20*m20)*2.0-d0*d2*m00*m20*(m21*m21)*2.0+d0*d2*(m10*m10)*m11*m20*2.0-d0*d1*m10*m11*(m22*m22)*2.0+d1*d2*m00*(m20*m20)*m21*2.0-d1*d2*(m00*m00)*m21*m22*2.0+d0*d1*(m10*m10)*m20*m21*2.0+d0*d1*m10*(m20*m20)*m22*2.0+d0*d2*m10*(m20*m20)*m21*2.0+d0*d1*m10*(m21*m21)*m22*2.0+d0*d2*m11*m20*(m21*m21)*2.0-d0*d2*(m10*m10)*m20*m22*2.0+d1*d2*m10*m20*(m21*m21)*2.0-d0*d2*(m11*m11)*m20*m22*2.0-d1*d2*m11*(m20*m20)*m21*2.0-d1*d2*(m10*m10)*m21*m22*2.0-(d1*d1)*m00*m10*m20*m21*2.0-(d2*d2)*m00*m10*m20*m21*2.0-(d0*d0)*m10*m11*m20*m21*2.0-(d2*d2)*m10*m11*m20*m21*2.0-(d0*d0)*m10*m20*m21*m22*2.0-(d1*d1)*m10*m20*m21*m22*2.0+d0*d2*m00*m10*m11*m21*2.0+d1*d2*m00*m10*m11*m20*2.0+d0*d1*m00*m11*m20*m21*2.0+d0*d2*m00*m10*m21*m22*2.0+d1*d2*m00*m10*m20*m22*2.0+d0*d1*m00*m20*m21*m22*2.0+d0*d2*m10*m11*m21*m22*2.0+d1*d2*m10*m11*m20*m22*2.0+d0*d1*m11*m20*m21*m22*2.0-(g*g)*m00*m10*m11*m20*m21*m22*4.0);

    VectorXd pf(6);
    pf = coeff.tail(6); // t^5, ... t^0
    pf = -pf / coeff(0);
    // MatrixXd P = MatrixXd::Zero(6, 6);
    MatrixXd P{6, 6};
    P.setZero();
    P.row(0) = pf.transpose();
    P.bottomLeftCorner(5, 5) = MatrixXd::Identity(5, 5);
    std::cout << "matP: \n" << P << '\n';
    Eigen::EigenSolver<Eigen::MatrixXd> es(P);
    MatrixXcd evals = es.eigenvalues(); // note: MatrixXcd
    std::cout << "Eigen Method Roots:\n" << evals.transpose() << std::endl;

    // Check results
    Eigen::PolynomialSolver<double, 6> solve;
    solve.compute(coeff.reverse()); // t^0,...,t^6
    std::cout << "PolynomialSolver Roots:\n" << solve.roots().transpose() << endl;

    MatrixXd evalsReal = evals.real();
    MatrixXd evalsImag = evals.imag();
    std::set<double> roots;
    for(size_t i = 0; i < evalsImag.rows(); ++i){
        //TODOs, change to fabs
        if(evalsImag(i) == 0) roots.emplace(evalsReal(i));
    }
    if(!roots.empty()){
        lambda = *roots.begin();
        std::cout << "find Lagrange multiplier: " << lambda << '\n';
        return true;
    } else {
        std::cout << "find Lagrange multiplier failed." << '\n';
        return false;
    }
}

bool StereoAlignmentGravityNorm(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x_){
    const int N = all_image_frame.size();
    const int n_r = 6*(N-1);
    const int n_state = 3*N+3;
    MatrixXd A1{n_r, 3*N};
    A1.setZero();
    MatrixXd A2{n_r, 3};
    A2.setZero();
    VectorXd b{n_r};
    b.setZero();
    x_.resize(n_state);

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 9);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        // tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0]
                                - frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T);
        // cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        // cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        A1.block<6, 6>(i * 6, i * 3) += tmp_A.leftCols<6>();
        A2.middleRows<6>(i * 6) += tmp_A.rightCols<3>();
        b.segment<6>(i * 6) += tmp_b;
    }
    A1 = A1 * 100.0;
    A2 = A2 * 100.0;
    b = b * 100.0;
    MatrixXd A1tA1 = A1.transpose() * A1;
    MatrixXd A1tA1_inv = A1tA1.inverse();
    MatrixXd I = MatrixXd::Identity(n_r,n_r);
    Matrix3d D = A2.transpose()*(I -A1*A1tA1_inv*A1.transpose())*A2;
    // std::cout << "matD: \n" << D << '\n';
    Vector3d d = A2.transpose()*(I -A1*A1tA1_inv*A1.transpose())*b;
    // std::cout << "vecd: \n" << d.transpose() << '\n';
    double lambda = 0.0;
    if(FindLagrange(D, d, lambda)){
        g = (D - MatrixXd::Identity(3,3)*lambda).inverse()*d;
        VectorXd x1 = -A1tA1_inv * A1.transpose() * A2 * g + A1tA1_inv * A1.transpose() * b;
        std::cout << "first gravity solved, norm: " << g.norm() << ", vector: " << g.transpose() << std::endl;
        if(fabs(g.norm() - G.norm()) > 0.5 || std::isnan(g.norm()))// || s < 0
        {
            return false;
        }
        x_.head(3*N) = x1;
        x_.tail(3) = g;
        return true;
    } else {
        return false;
    }
}

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    solveGyroscopeBias(all_image_frame, Bgs);

    if(LinearAlignment(all_image_frame, g, x))
        return true;
    else 
        return false;
}
