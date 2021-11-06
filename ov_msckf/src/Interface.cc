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

void InitSystem(const std::string &vins_config_file)
{

    // Create our VIO system
    sys = std::make_shared<VioManager>(params);
    // printf("config_file: %s\n", config_file.c_str());
    //for vins estimator
    //readParameters(params.path_vins_config);
    readParameters(vins_config_file);
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

void FeedImagesData(const ImagesInputData &data, const size_t num_cam)
{
    double ts = (double)(data.ts) * mTimescale;
    cv::Mat left = data.imgs[3];
    cv::Mat right = data.imgs[2];

    ov_core::CameraData message;
    message.timestamp = ts;

    if(num_cam == 5) {
        
        message.sensor_ids.push_back(2);
        message.images.push_back(data.imgs[0]);
        // message.gpu_imgs.emplace_back(data.imgs[0]);
        sys->feed_measurement_camera(message);

        message.sensor_ids[0] = 3;
        message.images[0] = data.imgs[1];
        // message.gpu_imgs.emplace_back(data.imgs[1]);
        sys->feed_measurement_camera(message);

        message.sensor_ids[0] = 4;
        message.images[0] = data.imgs[4];
        // message.gpu_imgs.emplace_back(data.imgs[4]);
        sys->feed_measurement_camera(message);
    }
    message.sensor_ids.clear();
    message.images.clear();
    // message.gpu_imgs.clear();

    message.sensor_ids.push_back(0);
    message.sensor_ids.push_back(1);
    message.images.push_back(left);
    // message.gpu_imgs.emplace_back(left);
    message.images.push_back(right);
    // message.gpu_imgs.emplace_back(left);
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
    if (!sys->initialized()) {
        result.valid = false;
        return;
    }
    else {
        result.valid = true;
    }
    auto pstate = sys->get_state();
    
    result.t = pstate->_imu->pos();
    result.R = quat_2_Rot(pstate->_imu->quat());
    result.v = pstate->_imu->vel();
    // result.gyro = ;
    // result.acc = ;
    result.ts = pstate->_timestamp;
    // result.ba.setZero();
    // result.bw.setZero();
    result.ba = pstate->_imu->bias_a();
    result.bw = pstate->_imu->bias_g();

    // Get our good features
    result.feats_msckf = sys->get_good_features_MSCKF();

    // Get our slam features
    result.feats_slam = sys->get_features_SLAM();

    // Get our ARUCO features
    result.feats_aruco = sys->get_features_ARUCO();

}

void ObtainLocalizationResult3(double ts, LocalizationOutputResult& result)
{
	if (!sys->initialized()) {
		result.valid = false;
		return;
	}
    else {
        result.valid = true;
    }
    auto pstate = sys->get_state();
    Eigen::Matrix<double, 13, 1> state_plus;
    sys->get_propagator()->fast_state_propagate(pstate, ts, state_plus);

	result.t = state_plus.block<3,1>(4,0);
	result.R = quat_2_Rot(state_plus.block<4, 1>(0, 0));
	result.v = state_plus.block<3, 1>(7, 0);
	result.gyro = state_plus.block<3, 1>(10, 0);
	// result.acc = ;
	result.ts = ts;
	// result.ba.setZero();
	// result.bw.setZero();
	result.ba = pstate->_imu->bias_a();
	result.bw = pstate->_imu->bias_g();

    // // Get our good features
    // result.feats_msckf = sys->get_good_features_MSCKF();

    // // Get our slam features
    // result.feats_slam = sys->get_features_SLAM();

    // // Get our ARUCO features
    // result.feats_aruco = sys->get_features_ARUCO();
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