#include "stdafx.h"
#include "ConvLayer.h"


// �������� ���������� ����
// ������������� ����� �� ������ Xavier
ConvLayer::ConvLayer(TensorSize size, int _nFilters, int _filterSize, int _nZeroesFilling, int _convolutionStep) : distribution(0.0, sqrt(2.0 / (_filterSize*_filterSize*size.depth)))
{
	// ���������� ������� ������
	inputSize.width  = size.width;
	inputSize.height = size.height;
	inputSize.depth  = size.depth;

	// ��������� �������� ������
	outputSize.width  = (size.width  - _filterSize + 2 * _nZeroesFilling) / _convolutionStep + 1;
	outputSize.height = (size.height - _filterSize + 2 * _nZeroesFilling) / _convolutionStep + 1;
	outputSize.depth  = _nFilters;

	this->nZeroesFilling  = _nZeroesFilling; // ��������� ���������� ������
	this->convolutionStep = _convolutionStep; // ��������� ��� ������

	this->nFilters    = _nFilters; // ��������� ����� ��������
	this->filterSize  = _filterSize; // ��������� ������ ��������
	this->filterDepth = size.depth; // ��������� ������� ��������

	// ��������� fc �������� ��� ����� �������� � �� ����������
	filters      = std::vector<Tensor>(_nFilters, Tensor(_filterSize, _filterSize, filterDepth));
	deltaFilters = std::vector<Tensor>(_nFilters, Tensor(_filterSize, _filterSize, filterDepth));

	// ��������� fc ����� ��� ����� �������� � �� ����������
	biases      = std::vector<double>(_nFilters, 0);
	deltaBiases = std::vector<double>(_nFilters, 0);

	InitWeights(); // �������������� ������� ������������
}
// ������������� ������� �������������
void ConvLayer::InitWeights()
{
	// ���������� �� ������� �� ��������
	for (int iFilter = 0; iFilter < nFilters; iFilter++)
	{
		for (int iFilterSize = 0; iFilterSize < filterSize; iFilterSize++)
		{
			for (int jFilterSize = 0; jFilterSize < filterSize; jFilterSize++)
			{
				for (int iDepth = 0; iDepth < filterDepth; iDepth++)
				{
					filters[iFilter](iDepth, iFilterSize, jFilterSize) = distribution(generator); // ���������� ��������� ����� � ���������� ��� � ������� �������
				}
			}
		}

		biases[iFilter] = 0.01; // ��� �������� ������������� � 0.01
	}
}
// ������ ���������������
Tensor ConvLayer::Forward(const Tensor &X)
{
	Tensor output(outputSize); // ������ �������� ������
							   // ���������� �� ������� �� ��������
	for (int iFilter = 0; iFilter < nFilters; iFilter++)
	{
		for (int iOutH = 0; iOutH < outputSize.height; iOutH++)
		{
			for (int iOutW = 0; iOutW < outputSize.width; iOutW++)
			{
				double outImageElement = biases[iFilter]; // ����� ���������� ��������
				// ���������� �� ��������
				for (int iFilterSize = 0; iFilterSize < filterSize; iFilterSize++)
				{
					for (int jFilterSize = 0; jFilterSize < filterSize; jFilterSize++)
					{
						const int i0 = convolutionStep * iOutH + iFilterSize - nZeroesFilling;
						const int j0 = convolutionStep * iOutW + jFilterSize - nZeroesFilling;

						// ��������� ��� ������ �������� ������� �������� �������, �� ������ ���������� ��
						if (i0 < 0 || i0 >= inputSize.height || j0 < 0 || j0 >= inputSize.width)
							continue;

						// ���������� �� ���� ������� ������� � ������� �����
						for (int iDepth = 0; iDepth < filterDepth; iDepth++)
						{
							outImageElement += X(iDepth, i0, j0) * filters[iFilter](iDepth, iFilterSize, jFilterSize);
						}
					}
				}
				/// ����� ���������� ��������. RELU
				outImageElement = outImageElement >= 0 ? outImageElement : 0.01*outImageElement;
				output(iFilter, iOutH, iOutW) = outImageElement;
			}
		}
	}

	return output; // ���������� �������� ������
}
// �������� ���������������
Tensor ConvLayer::Backward(const Tensor &dout, const Tensor &X)
{
	TensorSize size; // ������ �����

	// ����������� ������ ��� �����
	size.height = convolutionStep * (outputSize.height - 1) + 1;
	size.width  = convolutionStep * (outputSize.width - 1) + 1;
	size.depth  = outputSize.depth;

	Tensor deltas(size); // ������ ������ ��� �����

	// ����������� �������� �����
	for (int iDepth = 0; iDepth < size.depth; iDepth++)
	{
		for (int iOutH = 0; iOutH < outputSize.height; iOutH++)
		{
			for (int iOutW = 0; iOutW < outputSize.width; iOutW++)
			{
				deltas(iDepth, iOutH * convolutionStep, iOutW * convolutionStep) = dout(iDepth, iOutH, iOutW) * (X(iDepth, iOutH, iOutW) >= 0 ? 1.0 : 0.01);
			}
		}
	}

	// ����������� ��������� ����� �������� � ��������
	for (int iFilter = 0; iFilter < nFilters; iFilter++)
	{
		for (int iInputH = 0; iInputH < size.height; iInputH++)
		{
			for (int iInputW = 0; iInputW < size.width; iInputW++)
			{
				double delta = deltas(iFilter, iInputH, iInputW); // ���������� �������� ���������

				for (int iFilterSize = 0; iFilterSize < filterSize; iFilterSize++)
				{
					for (int jFilterSize = 0; jFilterSize < filterSize; jFilterSize++)
					{
						const int i0 = convolutionStep * iInputH + iFilterSize - nZeroesFilling;
						const int j0 = convolutionStep * iInputW + jFilterSize - nZeroesFilling;

						// ���������� ��������� �� ������� ��������
						if (i0 < 0 || i0 >= inputSize.height || j0 < 0 || j0 >= inputSize.width)
							continue;

						// ���������� �������� �������
						for (int iDepth = 0; iDepth < filterDepth; iDepth++)
						{
							deltaFilters[iFilter](iDepth, iFilterSize, jFilterSize) += delta * X(iDepth, i0, j0);
						}

					}
				}
				deltaBiases[iFilter] += delta; // ���������� �������� ��������
			}
		}
	}

	int pad = filterSize - 1 - nZeroesFilling; // �������� �������� ����������
	Tensor dX(inputSize); // ������ ������ ���������� �� �����

						  // ����������� �������� ���������
	for (int iInputH = 0; iInputH < inputSize.height; iInputH++)
	{
		for (int iInputW = 0; iInputW < inputSize.width; iInputW++)
		{
			for (int iInDepth = 0; iInDepth < filterDepth; iInDepth++)
			{
				double gradientElement = 0; // ����� ��� ���������

				// ��� �� ���� ������� ������������� ��������
				for (int iFilterSize = 0; iFilterSize < filterSize; iFilterSize++)
				{
					for (int jFilterSize = 0; jFilterSize < filterSize; jFilterSize++)
					{
						const int i0 = iInputH + iFilterSize - pad;
						const int j0 = iInputW + jFilterSize - pad;

						// ���������� ��������� �� ������� ��������
						if (i0 < 0 || i0 >= size.height || j0 < 0 || j0 >= size.width)
							continue;

						// ��������� �� ���� ��������
						for (int iFilter = 0; iFilter < nFilters; iFilter++)
						{
							gradientElement += filters[iFilter](iInDepth, filterSize - 1 - iFilterSize, filterSize - 1 - jFilterSize) * deltas(iFilter, i0, j0); // ��������� ������������ ��������� �������� �� ������
						}
					}
				}

				dX(iInDepth, iInputH, iInputW) = gradientElement; // ���������� ��������� � ������ ���������
			}
		}
	}

	return dX; // ���������� ������ ����������
}
// ���������� ������� �������������
void ConvLayer::UpdateWeights(const double learningRate)
{
	for (int iFilter = 0; iFilter < nFilters; iFilter++)
	{
		for (int iFilterSize = 0; iFilterSize < filterSize; iFilterSize++)
		{
			for (int jFilterSize = 0; jFilterSize < filterSize; jFilterSize++)
			{
				for (int iDepth = 0; iDepth < filterDepth; iDepth++)
				{
					filters[iFilter](iDepth, iFilterSize, jFilterSize) -= learningRate * deltaFilters[iFilter](iDepth, iFilterSize, jFilterSize); // �������� ��������, ���������� �� �������� ��������
					deltaFilters[iFilter](iDepth, iFilterSize, jFilterSize) = 0; // �������� �������� �������
				}
			}
		}

		biases[iFilter] -= learningRate * deltaBiases[iFilter]; // �������� ��������, ���������� �� �������� ��������
		deltaBiases[iFilter] = 0; // �������� �������� ���� ��������
	}
}
TensorSize ConvLayer::GetOutputSize()
{
	return this->outputSize;
}
