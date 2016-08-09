#include<Eigen/Dense>
#include<iostream>
using namespace Eigen;



class JointBayesian{
public:
	Matrix<double, Dynamic, Dynamic>A, G;//´æ´¢ÑµÁ·Ä£ĞÍ
	Matrix<double, 1, Dynamic>ratio;//´æ´¢²âÊÔ±ÈÂÊ
	JointBayesian(){};
	void jointbayesian_train(Matrix<double, Dynamic, Dynamic>trainset, Matrix<int, Dynamic, 1>trainlabel);
	void jointbayesian_test(Matrix<double, Dynamic, Dynamic>testset, Matrix<int, Dynamic, 1>testlabel, double threshold_start, double threshold_end, double step);
};