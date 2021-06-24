#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <memory>
#include <Eigen/Eigen>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <pangolin/pangolin.h>
#include <mutex>
#include <thread>
#include <GL/glut.h>
#include "Interface.h"

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"

#include "utils/dataset_reader.h"
#include "utils/parse_cmd.h"
#include "utils/sensor_data.h"

#include "estimator/parameters.h"

using namespace ov_msckf;
extern VioManagerOptions params;

enum SensorType
{
    Camera_Sensor,
    Imu_Sensor
};

Eigen::Matrix3d Rbc0, Rbc1;
Eigen::Vector3d tbc0, tbc1;

std::map<long long, LocalizationOutputResult> result_buffer;
std::mutex result_mtx;
std::map<int, Eigen::Vector3d> id_position_map;

bool resultEnd = false;
double MAX_DEPTH = 40.0, MIN_DEPTH = 0.1;

void drawCamera(double cam_width)
{
    const double cam_height = cam_width * 0.75;
    const double cam_depth = cam_width * 0.6;

    glLineWidth(2);

    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(cam_width, cam_height, cam_depth);
    glVertex3f(0, 0, 0);
    glVertex3f(cam_width, -cam_height, cam_depth);
    glVertex3f(0, 0, 0);
    glVertex3f(-cam_width, -cam_height, cam_depth);
    glVertex3f(0, 0, 0);
    glVertex3f(-cam_width, cam_height, cam_depth);

    glVertex3f(cam_width, cam_height, cam_depth);
    glVertex3f(cam_width, -cam_height, cam_depth);

    glVertex3f(-cam_width, cam_height, cam_depth);
    glVertex3f(-cam_width, -cam_height, cam_depth);

    glVertex3f(-cam_width, cam_height, cam_depth);
    glVertex3f(cam_width, cam_height, cam_depth);

    glVertex3f(-cam_width, -cam_height, cam_depth);
    glVertex3f(cam_width, -cam_height, cam_depth);

    glEnd();
}

void viewResult(const std::string &config_file)
{

    cv::FileStorage fs;
    fs.open(config_file, cv::FileStorage::READ);
    cv::Mat Tbc0, Tbc1;
    // fs["body_T_cam0"] >> Tbc0;
    // fs["body_T_cam1"] >> Tbc1;
    fs["T_C0toI"] >> Tbc0;
    fs["T_C1toI"] >> Tbc1;
    Eigen::Matrix4d T_bc0, T_bc1;
    cv::cv2eigen(Tbc0, T_bc0);
    cv::cv2eigen(Tbc1, T_bc1);

    pangolin::CreateWindowAndBind("vins_fusion: Viewer", 1280, 720);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", false, true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
    pangolin::Var<bool> menuShowGrid("menu.Show Grid", true, true);
    pangolin::Var<bool> menuShowHistoricalTrajectory("menu.Show Trajectory", true, true);
    pangolin::Var<bool> menuShowVelocityDir("menu.Show Velocity", false, true);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500.0, 500.0, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(-2, -2, 2, 0, 0, 0, 0.0, 0.0, 1.0));

    pangolin::Handler3D handler(s_cam);
    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        if (menuShowGrid)
        {
            double interval = 0.5;
            double Range = 40;
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> Nodes;
            for (auto i = -0.5 * Range / interval; i < 0.5 * Range / interval; i++)
            {
                Eigen::Vector3d node0(-0.5 * Range, i * interval, 0);
                Eigen::Vector3d node1(0.5 * Range, i * interval, 0);
                Eigen::Vector3d node2(i * interval, 0.5 * Range, 0);
                Eigen::Vector3d node3(i * interval, -0.5 * Range, 0);
                Nodes.push_back(std::make_pair(node0, node1));
                Nodes.push_back(std::make_pair(node2, node3));
            }
            glBegin(GL_LINES);
            glLineWidth(2);
            glColor3f(0, 0.5, 0);
            for (auto it = Nodes.begin(); it != Nodes.end(); it++)
            {
                glVertex3f(it->first[0], it->first[1], it->first[2]);
                glVertex3f(it->second[0], it->second[1], it->second[2]);
            }
            glEnd();
        }

        if (result_buffer.empty())
            continue;

        if (menuShowHistoricalTrajectory)
        {
            auto it_end = result_buffer.end();
            it_end = std::prev(it_end, 1);
            for (auto it = result_buffer.begin(); it != it_end; it++)
            {
                glColor3f(0.0, 1.0, 0.0);
                Eigen::Matrix4d Twb0 = Eigen::Matrix4d::Identity();
                Eigen::Matrix3d R_wb0 = it->second.R;
                Eigen::Vector3d t_wb0 = it->second.t;
                Twb0.topLeftCorner(3, 3) = R_wb0;
                Twb0.topRightCorner(3, 1) = t_wb0;
                Eigen::Matrix4d Twc0 = Twb0 * T_bc0;
                Eigen::Vector3d twc0 = Twc0.block(0, 3, 3, 1);

                auto itt = it;
                itt++;
                Eigen::Matrix4d Twb1 = Eigen::Matrix4d::Identity();
                Eigen::Matrix3d R_wb1 = itt->second.R;
                Eigen::Vector3d t_wb1 = itt->second.t;
                Twb1.topLeftCorner(3, 3) = R_wb1;
                Twb1.topRightCorner(3, 1) = t_wb1;
                Eigen::Matrix4d Twc1 = Twb1 * T_bc0;
                Eigen::Vector3d twc1 = Twc1.block(0, 3, 3, 1);

                glColor3f(1.0, 0.0, 1.0);
                glLineWidth(5);
                glPointSize(10);
                glBegin(GL_LINES);

                Eigen::Vector3d twb1 = itt->second.t;
                glVertex3f(twc0[0], twc0[1], twc0[2]);
                glVertex3f(twc1[0], twc1[1], twc1[2]);

                glEnd();
            }
        }

        Eigen::Matrix4d Twb = Eigen::Matrix4d::Identity();
        Eigen::Matrix3d Rwb = result_buffer.rbegin()->second.R;
        Eigen::Vector3d twb = result_buffer.rbegin()->second.t;
        Twb.topLeftCorner(3, 3) = Rwb;
        Twb.topRightCorner(3, 1) = twb;
        Eigen::Matrix4d Twc0 = Twb * T_bc0;
        glPushMatrix();
        glMultMatrixd(Twc0.data());
        glColor3f(1.0, 1.0, 0.0);
        drawCamera(0.05);
        glPopMatrix();

        if (menuShowVelocityDir)
        {
            glBegin(GL_LINES);
            Eigen::Vector3d v0 = Twc0.block(0, 3, 3, 1);
            Eigen::Vector3d velocity = result_buffer.rbegin()->second.v;
            double scale = 0.5;
            Eigen::Vector3d v1 = v0 + scale * velocity;
            glColor3f(0.0, 1.0, 0.0);
            glVertex3f(v0[0], v0[1], v0[2]);
            glVertex3f(v1[0], v1[1], v1[2]);
            glEnd();
        }

        if(menuShowPoints)
        {
            std::vector<std::vector<Eigen::Matrix<double, 6, 1>>> landmarks = result_buffer.rbegin()->second.landmarks;

            for(size_t i = 0; i < landmarks.size(); i++)
            {
                if(landmarks[i].empty())
                    continue;
                switch (i) {
                    case 0:
                        glColor3f(1.0, 0.0, 0.0); // red
                        break;
                    case 1:
                        glColor3f(1.0, 0.0, 0.0);
                        break;
                    case 2:
                        glColor3f(0.0, 0.0, 1.0); // blue
                        break;
                    case 3:
                        glColor3f(1.0, 0.6, 0.2); // orange
                        break;
                    case 4:
                        glColor3f(0.0, 1.0, 0.0); //green
                        break;
                }
                glBegin(GL_POINTS);
                for(size_t j = 0; j < landmarks[i].size(); j++)
                {
                    double depth = landmarks[i][j][5];
                    if(depth >= MIN_DEPTH && depth <= MAX_DEPTH)
                    {
                        glVertex3f(landmarks[i][j][0], landmarks[i][j][1], landmarks[i][j][2]);
                    }

                }
                glEnd();
            }
            glColor3f(0.9, 1.0, 1.0);
            for(size_t i = 0; i < landmarks.size(); i++)
            {
                if (landmarks[i].empty())
                    continue;
                for(size_t j = 0; j < landmarks[i].size(); j++)
                {
                    // std::cout << "id: " << landmarks[i][j][3] << " position: " <<  landmarks[i][j].head(3).transpose() << std::endl;

                    int id = landmarks[i][j][3];
                    double depth = landmarks[i][j][5];
                    if(depth < MIN_DEPTH || depth > MAX_DEPTH)
                        continue;
                    if(!id_position_map.empty()) {
                        auto it = id_position_map.find(id);
                        if(it != id_position_map.end())
                            pangolin::glDrawCross(landmarks[i][j][0], landmarks[i][j][1], landmarks[i][j][2], 0.01 * landmarks[i][j][4]);
                    }
                }
            }
            // std::cout << "_______________________\n";
            // lm << landmark[0], landmark[1], landmark[2], it.feature_id, it.feature_per_frame.size(), it.estimated_depth;
            

            id_position_map.clear();
            for(size_t i = 0; i < landmarks.size(); i++)
            {
                if (landmarks[i].empty())
                    continue;
                for(size_t j = 0; j < landmarks[i].size(); j++)
                {
                    int id = landmarks[i][j][3];
                    id_position_map[id] = landmarks[i][j].head(3);
                }
            }
        }

        if (menuFollowCamera)
        {
            pangolin::OpenGlMatrix Twc_gl;

            Twc_gl.m[0] = Twc0(0, 0);
            Twc_gl.m[1] = Twc0(1, 0);
            Twc_gl.m[2] = Twc0(2, 0);
            Twc_gl.m[3] = 0.0;

            Twc_gl.m[4] = Twc0(0, 1);
            Twc_gl.m[5] = Twc0(1, 1);
            Twc_gl.m[6] = Twc0(2, 1);
            Twc_gl.m[7] = 0.0;

            Twc_gl.m[8] = Twc0(0, 2);
            Twc_gl.m[9] = Twc0(1, 2);
            Twc_gl.m[10] = Twc0(2, 2);
            Twc_gl.m[11] = 0.0;

            Twc_gl.m[12] = Twc0(0, 3);
            Twc_gl.m[13] = Twc0(1, 3);
            Twc_gl.m[14] = Twc0(2, 3);
            Twc_gl.m[15] = 1.0;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024, 768, 500.0, 500.0, 512, 389, 0.1, 1000));
            s_cam.Follow(Twc_gl);
        }
        pangolin::FinishFrame();
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << "Usage: ./bin config_path basePath\n";
        return 0;
    }

    std::string configPath = argv[1];
    std::string basePath = argv[2];
    std::string imu_file = basePath + "/imu.txt";
    std::string association_file = basePath + "/images/association.txt";
    std::string seq_file = basePath + "/seq.txt";
    
    // Read in our openvins parameters
    // params = parse_command_line_arguments(argc, argv);
    params = parse_ov(argv[1]);
    InitSystem(configPath);
    size_t num = 5;
    std::map<long long, std::vector<SensorType>> data_sequence;
    std::map<long long, std::vector<std::string>> imgs_str_buffer;
    std::map<long long, std::pair<Eigen::Vector3d, Eigen::Vector3d>> imu_buffer;

    std::fstream f_association;
    f_association.open(association_file, std::ios::in);
    std::string line_association;
    while (getline(f_association, line_association))
    {
        std::string ts_str;
        long long ts;
        std::stringstream ss(line_association);
        for (size_t i = 0; i < num; i++)
        {
            std::string imgPath;
            ss >> imgPath;
            if (ts_str.empty())
            {
                int n = imgPath.find_last_of("/");
                int m = imgPath.find_last_of(".");
                ts_str = imgPath.substr(n + 1, m - n);
                ts = std::atoll(ts_str.c_str());
            }
            imgs_str_buffer[ts].push_back(basePath + imgPath);
        }
        data_sequence[ts].push_back(Camera_Sensor);
    }
    f_association.close();

    std::fstream f_imu;
    f_imu.open(imu_file, std::ios::in);
    std::string line_imu;
    while (getline(f_imu, line_imu))
    {
        long long ts;
        std::stringstream ss(line_imu);
        double gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z;
        ss >> ts >> gyro_x >> gyro_y >> gyro_z >> acc_x >> acc_y >> acc_z;
        imu_buffer[ts] = std::make_pair(Eigen::Vector3d(acc_x, acc_y, acc_z), Eigen::Vector3d(gyro_x, gyro_y, gyro_z));
        data_sequence[ts].push_back(Imu_Sensor);
    }
    f_imu.close();

    cv::FileStorage fs;
    fs.open(configPath, cv::FileStorage::READ);
    int showPangolin = fs["show_pangolin"];
    fs.release();
    if (showPangolin)
    {
        std::thread viewer_th = std::thread(&viewResult, std::ref(configPath));
        viewer_th.detach();
    }

    std::map<long long, std::string> ts_file_map;

    std::fstream f_seq;
    f_seq.open(seq_file, std::ios::in);
    std::string line_seq;
    while (getline(f_seq, line_seq))
    {
        int dataId;
        int dataType;
        long long timestamp;
        std::stringstream ss(line_seq);
        ss >> dataId >> dataType >> timestamp;
        if(fabs((timestamp - imgs_str_buffer.rbegin()->first) * 1e-6) < 0.2)
            resultEnd = true;

        if (dataType == 0)
        {
            
            if(imu_buffer.find(timestamp) == imu_buffer.end())
                continue;
            ImuInputData data;
            data.ts = timestamp;
            data.acc = imu_buffer[timestamp].first;
            data.gyro = imu_buffer[timestamp].second;
            FeedImuData(data);
            LocalizationOutputResult result;
            // ObtainLocalizationResult(timestamp, result);
            ObtainLocalizationResult2(result);
            if(result.valid)
            {
                std::unique_lock<std::mutex> lock(result_mtx);
                {
                    if(!resultEnd)
                        result_buffer[timestamp] = result;
                }
            }
        }
        
        if (dataType == 13 || dataType == 3)
        {

            if(imgs_str_buffer.find(timestamp) == imgs_str_buffer.end())
                continue;
            ImagesInputData data;
            data.ts = timestamp;
            for(int i = 0; i < num; i++)
                data.imgs.push_back(cv::imread(imgs_str_buffer[timestamp][i], 0));
            double timeScale = 1e-6;
            double dts = (double)(timestamp) * timeScale;
            long long ts_ = (long long)(1e6 * dts);
            ts_file_map[ts_] = imgs_str_buffer[timestamp][3];

            FeedImagesData(data);
        }
    }
    std::cout << "association Runned Out\n";
    std::ofstream f_out;
    f_out.open("ts_file_map.txt", std::ios::out);
    for(auto it = ts_file_map.begin(); it != ts_file_map.end(); it++)
        f_out << it->first << " " << it->second << std::endl;
    f_out.close();
    // if(MODE == MAPPING)
    //     ExportMap();
    pause();

    return 0;
}