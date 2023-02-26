#pragma once

#include <vector>
#include <iostream>

using namespace std;

// ����������� �������
struct TensorSize
{
	int depth; // �������
	int height; // ������
	int width; // ������
	TensorSize() {};
	TensorSize(int d, int h, int w) : depth(d), height(h), width(w) {};
};

// ������
class Tensor
{
public:
	/// �������� �������
	void Init(int width, int height, int depth);
	/// �������� ��������� �������
	void SetValues(double *val);
	/// ������� ��������� �������
	void GetValues(double *val);
	/// �������� ������� �� �����������
	Tensor(int width, int height, int depth);
	/// �������� ������� �� �����������
	Tensor(const TensorSize &size); // �������� �� �������
	/// ������������� �������� ����������
	double& operator()(int iDepth, int iHeight, int iWidth);
	/// ������������� �������� ����������
	double operator()(int iDepth, int iHeight, int iWidth) const;
	/// �������� ���� ��������
	Tensor operator+(const  Tensor &other);
	/// �������� ���� ��������
	Tensor operator-(const  Tensor &other);
	/// ��������� ��������� ������� �� �����
	Tensor operator*(const double alpha);
	/// ��������� �������� �������
	TensorSize GetSize() const;
	/// ������ ��������� ������� � �������
	friend std::ostream& operator<<(std::ostream& os, const Tensor &tensor);

private:
	/// ����������� �������
	TensorSize size;
	/// �������� ��������� �������
	std::vector<double> values;
	/// �������� ������ ��� ����������. �.�. ���������� ������ �������� � ����������
	int dw;

};
