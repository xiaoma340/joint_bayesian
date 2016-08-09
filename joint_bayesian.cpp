#include "joint_bayesian.h"



void JointBayesian::jointbayesian_train(Matrix<double, Dynamic, Dynamic>dataset, Matrix<int, Dynamic, 1>label){
	int n_dim = dataset.cols();//图片维数
	int n_image = dataset.rows();//图片数量
	int everyclasscount[10000] = { 0 };//记录每种类别的图片数
	int n_class = 1, cnt = 1;//n_class：种类数 cnt：每种种类的数量
	for (int i = 0; i < n_image; i++){//计算图片种类数（即有多少个不同的人）以及每种图片的个数。
		if (label(i, 0) != label(i + 1, 0)){
			everyclasscount[n_class - 1] = cnt;
			n_class++;
			cnt = 1;
			continue;
		}
		cnt++;
	}
	everyclasscount[n_class - 1] = cnt + 1;//最后一个种类的图片数。
	Matrix<double, Dynamic, Dynamic>u,e;
	u.setZero(n_dim, n_class);
	e.setZero(n_dim, n_image);
	Matrix<double, Dynamic, Dynamic>ui,ei;
	ui.setZero(n_dim, 1);
	ei.setZero(n_dim, 1);
	for (int i = 0; i < n_class; i++){
		for (int j = 0; j < everyclasscount[i]; j++)ui += dataset.row(i + j).transpose();
		ui = ui / everyclasscount[i];
		u.col(i) = ui;
		for (int j = 0; j < everyclasscount[i]; j++)e.col(i+j)= dataset.row(i + j).transpose()-ui;
	}
	Matrix<double, Dynamic, Dynamic>Su, Sw,oldSw;
	Su.setZero(n_dim,n_dim);
	Sw.setZero(n_dim, n_dim);
	for (int i = 0; i < n_class; i++){//初始化Su，Sw为正定矩阵
		Su += u.col(i)*u.col(i).transpose();
		for (int j = 0; j < everyclasscount[i]; j++)Sw += e.col(i + j)*e.col(i + j).transpose();
	}
	Su = Su / n_class;
	Sw = Sw / n_image;
	double convergence = 1, min_convergence = 1;
	Matrix<double, Dynamic, Dynamic>F;
	for (int l = 0; l < 500; l++){//开始迭代
		F = Sw.inverse();
		for (int i = 0; i < n_class; i++){
			G = -((everyclasscount[i] * Su + Sw).inverse())*Su*(Sw.inverse());
			ui.setZero(n_dim, 1);
			for (int j = 0; j < everyclasscount[i]; j++){
				ui += Su*(F + j*G)*(dataset.row(i + j).transpose());
				ei += Sw*G*(dataset.row(i + j).transpose());
				e.col(i + j) = dataset.row(i + j).transpose() + ei;
			}
		}
		for (int i = 0; i < n_class; i++){//计算Su，Sw
			Su += u.col(i)*u.col(i).transpose();
			for (int j = 0; j < everyclasscount[i]; j++)Sw += e.col(i + j)*e.col(i + j).transpose();
		}
		Su = Su / n_class;
		Sw = Sw / n_image;
		convergence = (Sw - oldSw).norm() / Sw.norm();
		if (convergence < 0.000001)break;
	}
	F = Sw.inverse();
	G = -((2 * Su + Sw).inverse())*Su*(Sw.inverse());
	A = ((Su + Sw).inverse()) - (F + G);
}


void JointBayesian::jointbayesian_test(Matrix<double, Dynamic, Dynamic>dataset, Matrix<int, Dynamic, 1>label, double t_s, double t_e, double step){
	int n_pair = dataset.rows() / 2;
	ratio.setZero(1, n_pair);
	Matrix<double, 1, 1>res;
	for (int i = 0; i < n_pair; i++){
		res = dataset.row(i)*A*dataset.row(i).transpose() + dataset.row(i+1)*A*dataset.row(i+1).transpose() - 2 * dataset.row(i)*G*dataset.row(i + 1).transpose();
		ratio(0, i) = res(0, 0);
	}
	double accuracy = 0, bestaccuracy = 0, bestthreshold = t_s;
	for (double z = t_s; z <= t_e; z += step){
		int score = 0, y;
		for (int j = 0; j < n_pair; j++){
			if (ratio(0, j) >= z)y = 1;
			else y = 0;
			if (label(j, 0) == y)score++;
		}
		accuracy = score / n_pair;
		if (accuracy > bestaccuracy){
			bestaccuracy = accuracy;
			bestthreshold = z;
		}
	}
	std::cout << "best threshold" << bestthreshold << std::endl;
	std::cout << "best accuracy" << bestaccuracy;
}

