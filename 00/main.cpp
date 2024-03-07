#include<cmath>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Dense>
#include<iostream>
#include<cmath>
int main(){

    Eigen::Vector2f src(2,1);
    Eigen::Matrix2f transform;
    float sqrt2=sqrt(2.0);
    transform << sqrt2/2, -sqrt2/2, 
                 sqrt2/2, sqrt2/2;
    std::cout << transform << std::endl;
    Eigen::Vector2f result = transform * src;
    std::cout << result << std::endl;
    result += Eigen::Vector2f(1,2);
    std::cout << result << std::endl;
    return 0;
}