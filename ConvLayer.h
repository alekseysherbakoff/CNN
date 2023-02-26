#pragma once
#include "Tensor.h"

#include <random>

using namespace std;

class ConvLayer
{
public:
	/// �������� ����������� ����
	ConvLayer(TensorSize size, int _nFilters, int _filterSize, int _nZeroesFilling, int _convolutionStep);
	/// ��������� ������� ������
	TensorSize GetOutputSize();
	/// ���������� ������� � �����������. ������ ���
	Tensor Forward(const Tensor &X);
	/// �������� ���. ���������� ��������� � �������. ���������� ������ ������ ��� ���������� ����� ������� � �������
	Tensor Backward(const Tensor &dout, const Tensor &X);
	/// ���������� �������
	void UpdateWeights(const double learningRate);

private:
	///������������� �������
	void InitWeights(); 

private:
	/// ��������� ��������� �����
	std::default_random_engine generator;
	/// ��������� ��������� ����� �� ����������� ������������� ������
	std::normal_distribution<double> distribution;
	///������ ����� ����������� ���� 
	TensorSize inputSize;
	/// ������ ������ ����������� ����
	TensorSize outputSize;
	/// ������
	std::vector<Tensor> filters;
	/// ��������
	std::vector<double> biases;
	/// �������� ��������� ��������
	std::vector<Tensor> deltaFilters;
	/// �������� ��������� ��������
	std::vector<double> deltaBiases;
	/// ���������� ������
	int nZeroesFilling;
	/// ��� �������
	int convolutionStep;
	/// ���������� ��������. ��� Mnist 1. �.�. ����������� �� � RGB, � � ����� �����
	int nFilters;
	/// ������ �������
	int filterSize;
	/// ������� ��������
	int filterDepth;
};