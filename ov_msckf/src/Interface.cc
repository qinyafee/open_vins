//
// Created by yukan on 2020/10/15.
//

#include "Interface.h"
#include <csignal>
#include <memory>

#include <mutex>

#include "core/VioManager.h"
// #include "core/VioManagerOptions.h"
// #include "utils/dataset_reader.h"
// #include "utils/parse_cmd.h"
#include "utils/sensor_data.h"

#include "estimator/parameters.h"

using namespace ov_msckf;

std::shared_ptr<VioManager> sys;
VioManagerOptions params;

double mTimescale;


Eigen::Vector3d t_drift_new, t_drift_old;
Eigen::Matrix3d r_drift_new, r_drift_old;
Eigen::Vector3d delta_t;
Eigen::AngleAxisd delta_r;
int correction_index;
int loop_closure_interval;
bool new_drift;
Eigen::Vector3d t_correction;
Eigen::Matrix3d R_correction;


int enable_loop;


//cv::Mat stable_img_left;
//cv::Mat stable_img_right;
//bool isStable;

void InitSystem(const std::string &config_file)
{

    // Create our VIO system
    sys = std::make_shared<VioManager>(params);
    // printf("config_file: %s\n", config_file.c_str());
    //for vins estimator
    // readParameters(config_file);
    // readParameters(params.path_vins_config);
    readParameters("/home/qyf/workspace/catkin_ws_ov/src/open_vins/config/Pimax/20210318.yaml");
    sys->vins_estimator.setParameter();

    mTimescale = 1e-6;
    t_drift_new.setZero();
    t_drift_old.setZero();
    r_drift_new.setIdentity();
    r_drift_old.setIdentity();
    int correction_index = 0;
    new_drift = false;

    // cv::FileStorage fs;
    // fs.open(config_file, cv::FileStorage::READ);
    // loop_closure_interval = fs["loop_interval"];
    // enable_loop = fs["enable_loop"];
    // fs.release();

//    isStable = false;
}

void FeedImagesData(const ImagesInputData &data)
{
    double ts = (double)(data.ts) * mTimescale;
    cv::Mat left = data.imgs[3];
    cv::Mat right = data.imgs[2];

    ov_core::CameraData message;
    message.timestamp = ts;
    message.sensor_ids.push_back(0);
    message.sensor_ids.push_back(1);
    message.images.push_back(left);
    message.images.push_back(right);
    sys->feed_measurement_camera(message);//only add to queue

/*    std::vector<cv::Mat> imgs = { left, right, data.imgs[0], data.imgs[1], data.imgs[4] };
    std::vector<cv::Mat> imgs = { left, right };
    std::vector<cv::Mat> masks;
    if (!data.masks.empty()) {
        //masks = { data.masks[3], data.masks[2], data.masks[0], data.masks[1], data.masks[4] };
        masks = { data.masks[3], data.masks[2]};
    }
    else {
        masks.resize(imgs.size());
    }
//    if(isStable)
//    {
//        imgs[0] = stable_img_left.clone();
//        imgs[1] = stable_img_right.clone();
//    }
    mEstimator->inputImage(ts, imgs, masks);
//    if(mEstimator->solver_flag == Estimator::NON_LINEAR && isStable == false)
//    {
//        isStable = true;
//        stable_img_left = left;
//        stable_img_right = right;
//    }

    if(enable_loop)
    {
        Eigen::Vector3d t_drift_cur = mEstimator->t_drift();
        Eigen::Matrix3d r_drift_cur = mEstimator->r_drift();

        double diff_t = fabs(t_drift_old.dot(t_drift_cur));
        Eigen::Matrix3d diff_R = r_drift_cur.transpose() * r_drift_old;
        Eigen::AngleAxisd diff_axang;
        diff_axang.fromRotationMatrix(diff_R);
        double diff_rotation = diff_axang.angle();
        if(diff_t > 1e-7 && diff_rotation > 1e-7)
        {
            t_drift_new = t_drift_cur;
            r_drift_new = r_drift_cur;
            new_drift = true;
            correction_index = 0;
        }


        Eigen::AngleAxisd delta_axang;
        delta_axang.fromRotationMatrix(r_drift_old.transpose() * r_drift_new);
        delta_t = (t_drift_cur - t_drift_old) / (double)(loop_closure_interval - 10);
        delta_r = Eigen::AngleAxisd(delta_axang.angle() / (double)(loop_closure_interval - 10), delta_axang.axis());
    }*/

}

void FeedImuData(const ImuInputData &data)
{
    double ts = (double)(data.ts) * mTimescale;
    // convert into correct format
    ov_core::ImuData message;
    message.timestamp = ts;
    message.wm = data.gyro;
    message.am = data.acc;
    // send it to our VIO system
    sys->feed_measurement_imu(message);//call track_image_and_update()    

    // mEstimator->inputIMU(ts, data.acc, data.gyro);

/*    if(mEstimator->solver_flag == Estimator::NON_LINEAR)
    {
        if(enable_loop)
        {
            if(new_drift)
            {
                t_correction = delta_t * correction_index + t_drift_old;
                R_correction = Eigen::AngleAxisd(delta_r.angle() * correction_index, delta_r.axis()).toRotationMatrix() * r_drift_old;
            }
            else
            {
                t_correction = t_drift_old;
                R_correction = r_drift_old;
            }
            
            correction_index++;
            if(correction_index == loop_closure_interval - 10)
            {
                t_drift_old = t_drift_new;
                r_drift_old = r_drift_new;
                new_drift = false;
            }
        }
    }*/
}


void ObtainLocalizationResult2(LocalizationOutputResult &result){
    auto pstate = sys->get_state();
    
    result.t = pstate->_imu->pos();
    result.R = quat_2_Rot(pstate->_imu->quat());
    // result.v = ;
    // result.gyro = ;
    // result.acc = ;
    result.ts = pstate->_timestamp;
    result.ba.setZero();
    result.bw.setZero();
}

/*
void ObtainLocalizationResult(long long timestamp, LocalizationOutputResult &result)
{
    double ts = (double)(timestamp) * mTimescale;
    auto lastestState = mEstimator->getLatestStateAt(ts);
    result.valid = lastestState.valid;
    if(result.valid)
    {
        result.landmarks = mEstimator->landmarks;
    }
    if (lastestState.valid) {
        if (enable_loop)
        {
            result.t = R_correction * lastestState.P + t_correction;
            result.R = R_correction * lastestState.Q;
            result.v = R_correction * lastestState.V;
        }
        else {
            result.t = lastestState.P;
            result.R = lastestState.Q;
            result.v = lastestState.V;
        }
        result.gyro = lastestState.gyro;
        result.acc = lastestState.acc;
        result.ts = ts;
        result.ba.setZero();
        result.bw.setZero();
    }
}


void ExportMap()
{
    mEstimator->exportMap();
}

std::vector<std::pair<int, int>> ObtainMatchingWithMap(long long timestamp)
{
    double ts = (double)(timestamp) * 1e-6;
    std::vector<std::pair<int, int>> matching = mEstimator->getMapPointMatching(ts);
    return matching;
}
*/