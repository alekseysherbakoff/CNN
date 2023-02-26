#pragma once

#include "stdafx.h"

#include <iostream>
#include <vector>

using namespace std;

class Matrix
{
	double** matrix;
	int row, col;
public:
	void Init(int row, int col);
	void Rand();
	Matrix MatMult(const Matrix& m1, const Matrix& m2);
	Matrix Transpose(const Matrix& m);
	static void VectorsSum(double* a, const double* b, int n);
	static void MatMult(const Matrix& m, const double* b, double* c);
	static void MatMultTranspose(const Matrix& m, const double* b, double* c);
	double& operator ()(int i, int j);
	friend ostream& operator << (ostream& os, const Matrix& m);
	friend istream& operator >> (istream& is, Matrix& m);
};

