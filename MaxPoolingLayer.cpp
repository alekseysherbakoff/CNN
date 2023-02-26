#include "stdafx.h"
#include "MaxPoolingLayer.h"
#include "Tensor.h"

// �������� ����
MaxPoolingLayer::MaxPoolingLayer(TensorSize size, int _scale) : mask(size) 
{
	// ���������� ������� ������
	inputSize.width = size.width;
	inputSize.height = size.height;
	inputSize.depth = size.depth;

	// ��������� �������� ������
	outputSize.width = size.width / _scale;
	outputSize.height = size.height / _scale;
	outputSize.depth = size.depth;

	this->scale = _scale; // ���������� ����������� ����������
}

// ������ ��������������� � �������������� �����
Tensor MaxPoolingLayer::Forward(const Tensor &X) 
{
	Tensor output(outputSize); // ������ �������� ������

	// ���������� �� ������� �� �������
	for (int iDepth = 0; iDepth < inputSize.depth; iDepth++) 
	{
		for (int iInputH = 0; iInputH < inputSize.height; iInputH += scale) 
		{
			for (int iInputW = 0; iInputW < inputSize.width; iInputW += scale) 
			{
				int imax = iInputH; // ������ ������ ���������
				int jmax = iInputW; // ������ ������� ���������
				double max = X(iDepth, iInputH, iInputW); // ��������� �������� ��������� - �������� ������ ������ ����������
				const int i0 = iInputH / scale;
				const int j0 = iInputW / scale;
				if (i0 < 0 || i0 >= outputSize.height || j0 < 0 || j0 >= outputSize.width)
					continue;
				// ���������� �� ���������� � ���� �������� � ��� ����������
				for (int y = iInputH; y < iInputH + scale; y++) 
				{
					for (int x = iInputW; x < iInputW + scale; x++) 
					{
						double value = X(iDepth, y, x); // �������� �������� �������� �������
						mask(iDepth, y, x) = 0; // �������� �����

						// ���� ������� �������� ������ �������������
						if (value > max) 
						{
							max = value; // ��������� ��������
							imax = y; // ��������� ������ ������ ���������
							jmax = x; // ��������� ������ ������� ���������
						}
					}
				}
				output(iDepth, i0, j0) = max; // ���������� � �������� ������ ��������� ��������
				mask(iDepth, imax, jmax) = 1; // ���������� 1 � ����� � ����� ������������ ������������� ��������
			}
		}
	}

	return output; // ���������� �������� ������
}

// �������� ���������������
Tensor MaxPoolingLayer::Backward(const Tensor &dout, const Tensor &X) 
{
	Tensor dX(inputSize); // ������ ������ ��� ����������
	for (int iDepth = 0; iDepth < inputSize.depth; iDepth++)
	{
		for (int iInputH = 0; iInputH < inputSize.height; iInputH++)
		{
			for (int iInputW = 0; iInputW < inputSize.width; iInputW++)
			{
				dX(iDepth, iInputH, iInputW) = dout(iDepth, iInputH / scale, iInputW / scale) * mask(iDepth, iInputH, iInputW); // �������� ��������� �� �����
			}
		}
	}

	return dX; // ���������� ����������� ���������
}