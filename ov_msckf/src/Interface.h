//
// Created by yukan on 2020/10/15.
//

#ifndef PI_SLAM_INTERFACE_H
#define PI_SLAM_INTERFACE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>



struct ImagesInputData{
    long long ts;
    std::vector<cv::Mat> imgs;
    std::vector<cv::Mat> masks;
};

struct ImuInputData{
    long long ts;
    Eigen::Vector3d gyro;
    Eigen::Vector3d acc;
};


struct LocalizationOutputResult {
    double ts;
    Eigen::Vector3d t;//global
    Eigen::Matrix3d R;//global
    Eigen::Vector3d v;//global
    Eigen::Vector3d gyro;//local
    Eigen::Vector3d acc;//global
    Eigen::Vector3d ba;
    Eigen::Vector3d bw;
    std::vector<std::vector<Eigen::Matrix<double, 6, 1>>> landmarks;

    std::vector<Eigen::Vector3d> feats_msckf;
    std::vector<Eigen::Vector3d> feats_slam;
    std::vector<Eigen::Vector3d> feats_aruco;
    bool valid;
};


void InitSystem(const std::string &config_file);

void FeedImagesData(const ImagesInputData &data);

void FeedImuData(const ImuInputData &data);

void ObtainLocalizationResult(long long timestamp, LocalizationOutputResult &result);
void ObtainLocalizationResult2(LocalizationOutputResult &result);

std::vector<std::pair<int, int>> ObtainMatchingWithMap(long long timestamp);

void ExportMap();

#endif //PI_SLAM_INTERFACE_H
