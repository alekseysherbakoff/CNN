#pragma once

#include "Tensor.h"
class MaxPoolingLayer 
{
public:
	MaxPoolingLayer(TensorSize size, int scale = 2); // �������� ����
	Tensor Forward(const Tensor &X); // ������ ���������������
	Tensor Backward(const Tensor &dout, const Tensor &X); // �������� ���������������
	TensorSize GetInputSize() { return inputSize; };
	TensorSize GetOutputSize() { return outputSize; };

private:
	TensorSize inputSize; // ������ �����
	TensorSize outputSize; // ������ ������

	int scale; // �� ������� ��� ����������� �����������
	Tensor mask; // ����� ��� ����������
};