// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(int x, int y, const Vector3f* _v,
float size_x, float size_y)
{   
    // 该像素的中点
    float x_mid = x+size_x/2;
    float y_mid = y+size_y/2;
    int ge_cnt = 0; // cnt of >=0 
    int le_cnt = 0; // cnt of <=0
    Eigen::Vector2f CA(_v[0][0]-_v[2][0],_v[0][1]-_v[2][1]);
    Eigen::Vector2f AB(_v[1][0]-_v[0][0],_v[1][1]-_v[0][1]);
    Eigen::Vector2f BC(_v[2][0]-_v[1][0],_v[2][1]-_v[1][1]);
    Eigen::Vector2f AP(x_mid-_v[0][0],y_mid-_v[0][1]);
    Eigen::Vector2f BP(x_mid-_v[1][0],y_mid-_v[1][1]);
    Eigen::Vector2f CP(x_mid-_v[2][0],y_mid-_v[2][1]);
    auto cross_product = [](const Eigen::Vector2f& a,
    const Eigen::Vector2f& b){
        return a.x()*b.y()-a.y()*b.x();
    };
    float result[] = {
        cross_product(AB,AP),
        cross_product(BC,BP),
        cross_product(CA,CP)
    };
    for(auto num:result){
        ge_cnt += num >= 0;
        le_cnt += num <= 0;
    }
    return ge_cnt == 3 || le_cnt == 3;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            // vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        //rasterize_triangle_without_msaa(t);
        rasterize_triangle_with_msaa(t);
    }
}
void rst::rasterizer::rasterize_triangle_with_msaa(const Triangle& t) {
    auto v = t.toVector4();
    int x_left_bound = std::min({v[0][0],v[1][0],v[2][0]});
    int x_right_bound = std::max({v[0][0],v[1][0],v[2][0]});
    int y_left_bound = std::min({v[0][1],v[1][1],v[2][1]});
    int y_right_bound = std::max({v[0][1],v[1][1],v[2][1]});
    int n=4;
    int m=4;
    for(int x=x_left_bound;x<=x_right_bound;x++)
        for(int y=y_left_bound;y<=y_right_bound;y++){
            int blockinTriangle = 0;
            auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
            float z_interpolated = alpha * v[0].z() + beta * v[1].z() + gamma * v[2].z();
            blockinTriangle = MSAA(x, y, t.v, n, m, z_interpolated, t.getColor());
            if(blockinTriangle>0){
                int idx = get_index(x, y);
                Eigen::Vector3f col = (msaa_frame_buf[idx*4]
                +msaa_frame_buf[idx*4+1]
                +msaa_frame_buf[idx*4+2]
                +msaa_frame_buf[idx*4+3])/4.0;
                set_pixel(Eigen::Vector3f(x, y, z_interpolated),
                col);
            }
        }
}
int rst::rasterizer::MSAA(int x, int y, const Vector3f* _v, int n, int m, float z, Eigen::Vector3f color)
{
    float size_x = 1.0/n; // the size_x of every super sample pixel
    float size_y = 1.0/m;

    int blocksinTriangle = 0;
    for(int i=0; i<n; ++i) 
        for(int j=0; j<m; ++j) {
            if (z>msaa_depth_buf[get_index(x,y)*4 + i*2 + j] && insideTriangle(x+i*size_x, y+j*size_y, _v, size_x, size_y)) {
                msaa_depth_buf[get_index(x,y)*4 + i*2 +j] = z;
                msaa_frame_buf[get_index(x,y)*4 + i*2 +j] = color;
                blocksinTriangle ++;
            }
        }
    return blocksinTriangle;
}
//Screen space rasterization
void rst::rasterizer::rasterize_triangle_without_msaa(const Triangle& t) {
    auto v = t.toVector4();
    int x_left_bound = std::min({v[0][0],v[1][0],v[2][0]});
    int x_right_bound = std::max({v[0][0],v[1][0],v[2][0]});
    int y_left_bound = std::min({v[0][1],v[1][1],v[2][1]});
    int y_right_bound = std::max({v[0][1],v[1][1],v[2][1]});
    for(int x=x_left_bound;x<=x_right_bound;x++)
        for(int y=y_left_bound;y<=y_right_bound;y++){
            if (insideTriangle(x, y, t.v,1,1)) {
                // 求得中心坐标
                auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
                // float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                // float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
// 求得深度,这里三个w都为1,所以直接写
                float z_interpolated = alpha * v[0].z() + beta * v[1].z() + gamma * v[2].z();
// 也对
                // z_interpolated *= w_reciprocal;
                if (z_interpolated > depth_buf[get_index(x,y)]) {
                    depth_buf[get_index(x,y)] = z_interpolated;
                    set_pixel(Eigen::Vector3f(x, y, z_interpolated), t.getColor());
                }
            }
        }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(msaa_frame_buf.begin(),msaa_frame_buf.end(),
        Eigen::Vector3f{0,0,0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), -2);
        std::fill(msaa_depth_buf.begin(),msaa_depth_buf.end(),
        -2);
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    msaa_depth_buf.resize(16*w*h);
    msaa_frame_buf.resize(16*w*h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on