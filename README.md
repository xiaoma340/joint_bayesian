包含文件：jointbayesian.h,jointbayesian.cpp

实现了一个JointBayesian类，类中封装了两个函数

1. void jointbayesian_train(Matrix<double, Dynamic, Dynamic>trainset, Matrix<int, Dynamic, 1>trainlabel);
功能：用于训练模型
参数：trainset：
							训练数据集矩阵，每行为一张图片；
			trainlabel：
			        训练集标签，每张图片属于哪个人；
			      
2.void jointbayesian_test(Matrix<double, Dynamic, Dynamic>testset, Matrix<int, Dynamic, 1>testlabel, double threshold_start, double threshold_end, double step);

功能：测试训练模型的性能
参数：testset：
							测试集矩阵，每行为一张图片，每两张图片为一对；
			label：
			      测试集标签，1表示一对图片为同一人，0表示一对图片为不同的人；
			threshold_start：
			      阈值起始值
			threshold_end：
			      阈值结束值
			 step：
			      阈值步进大小。