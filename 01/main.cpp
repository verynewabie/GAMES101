#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0],
                 0, 1, 0, -eye_pos[1],
                 0, 0, 1, -eye_pos[2],
                 0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(Eigen::Vector3f axis, float rotation_angle){
    // axis归一化
    axis.normalize();
    Eigen::Matrix3f matrix_product;
    matrix_product << 0,-axis(2),axis(1),
                      axis(2),0,-axis(0),
                      -axis(1),axis(0),0;
    float radian_angle = rotation_angle * MY_PI / 180;
    Eigen::Matrix3f rotate = Eigen::Matrix3f::Identity() 
    + sin(radian_angle) * matrix_product + (1-cos(radian_angle))
    *matrix_product*matrix_product;
    Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
    result.block(0,0,3,3) << rotate;
    return result;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model;
    // 先转换为弧度
    float radian_angle = rotation_angle * MY_PI / 180;
    float cos_val = cos(radian_angle);
    float sin_val = sin(radian_angle);
    model << cos_val, -sin_val,0,0,
             sin_val,  cos_val,0,0,
             0,0,1,0,
             0,0,0,1;
    return model;
}
// 45 1 0.1 50 分别是角度，长宽比，近平面距离，远平面距离
Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    float t = tan((eye_fov*MY_PI/180)/2) * fabs(zNear);
    float r = aspect_ratio * t;
    float l = -r;
    float b = -t;
    float n = zNear;
    float f = zFar;
    Eigen::Matrix4f ortho_translate;
    Eigen::Matrix4f ortho_scale;
    Eigen::Matrix4f persp2ortho;
    persp2ortho << n,0,0,0,
                   0,n,0,0,
                   0,0,n+f,-n*f,
                   0,0,1,0;
    ortho_translate << 1,0,0,-(l+r)/2,
                       0,1,0,-(t+b)/2,
                       0,0,1,-(n+f)/2,
                       0,0,0,1;
    ortho_scale << 2/(r-l),0,0,0,
                   0,2/(t-b),0,0,
                   0,0,2/(n-f),0,
                   0,0,0,1;
    return ortho_scale*ortho_translate*persp2ortho;
}

int main(int argc, const char** argv)
{
    // 绕z轴逆时针旋转angle度
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        // string to float
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }
    // 光栅化器类
    rst::rasterizer r(700, 700);
    // 相机位置
    Eigen::Vector3f eye_pos = {0, 0, 5};
    // 三角形三点坐标
    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};
    // 索引
    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);
    //  按下的按键
    int key = 0;
    // 生成帧的数量
    int frame_count = 0;

    if (command_line) {
        // 初始化 frame_buf and depth_buf
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        // 三个变换矩阵
        r.set_model(get_model_matrix(Eigen::Vector3f(0,0,1),angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, -0.1, -50));
        // 光栅化
        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        // CV_32FC3 32表示一个通道占32位 F表示浮点型 C3表示RGB彩色图像(三通道)
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        // 8表示一个通道占8位，U表示无符号整型，convertTo就是改变图像的数据类型，且可以选择尺度缩放，这里是1代表不缩放
        image.convertTo(image, CV_8UC3, 1.0f);
        // 保存图像到文件中
        cv::imwrite(filename, image);

        return 0;
    }
    // Esc
    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(Eigen::Vector3f(0,0,1),angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, -0.1, -50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
