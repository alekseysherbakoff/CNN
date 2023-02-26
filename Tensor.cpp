#include "stdafx.h"
#include "Tensor.h"

// инициализация по размерам
void Tensor::Init(int width, int height, int depth)
{
	size.width = width; // запоминаем ширину
	size.height = height; // запоминаем высоту
	size.depth = depth; // запоминаем глубину
	dw = depth * width; // запоминаем произведение глубины на ширину для индексации
	values = std::vector<double>(width * height * depth, 0); // создаём вектор из width * height * depth нулей
}

// создание из размеров
Tensor::Tensor(int width, int height, int depth)
{
	Init(width, height, depth);
}

// создание из размера
Tensor::Tensor(const TensorSize &size)
{
	Init(size.width, size.height, size.depth);
}

// индексация
double& Tensor::operator()(int iDepth, int iH, int iW)
{
	return values[iH * dw + iW * size.depth + iDepth];
}

// индексация
double Tensor::operator()(int iDepth, int iH, int iW) const
{
	return values[iH * dw + iW * size.depth + iDepth];
}
Tensor Tensor::operator+(const  Tensor &other)
{
	for (int iDepth = 0; iDepth < size.depth; iDepth++)
	{
		for (int iH = 0; iH < size.height; iH++)
		{
			for (int iW = 0; iW < size.width; iW++)
			{
				values[iH * dw + iW * size.depth + iDepth] += other.values[iH * dw + iW * size.depth + iDepth];
			}
		}
	}

	return *this;
}

Tensor Tensor::operator-(const  Tensor &other)
{
	for (int iDepth = 0; iDepth < size.depth; iDepth++)
	{
		for (int iH = 0; iH < size.height; iH++)
		{
			for (int iW = 0; iW < size.width; iW++)
			{
				values[iH * dw + iW * size.depth + iDepth] -= other.values[iH * dw + iW * size.depth + iDepth];
			}
		}
	}

	return *this;
}
Tensor Tensor::operator*(const double alpha)
{
	for (int iDepth = 0; iDepth < size.depth; iDepth++)
	{
		for (int iH = 0; iH < size.height; iH++)
		{
			for (int iW = 0; iW < size.width; iW++)
			{
				values[iH * dw + iW * size.depth + iDepth] *= alpha;
			}
		}
	}
	return *this;
}
// получение размера
TensorSize Tensor::GetSize() const
{
	return size;
}

void Tensor::SetValues(double *in)
{
	const int tSize = size.depth* size.height * size.width;
	memcpy(&values[0], &in[0], sizeof(double)*tSize);
}
void Tensor::GetValues(double *out)
{
	const int tSize = size.depth* size.height * size.width;
	memcpy(&out[0], &values[0], sizeof(double)*tSize);
}

// вывод тензора
std::ostream& operator<<(std::ostream& os, const Tensor &tensor)
{
	for (int iDepth = 0; iDepth < tensor.size.depth; iDepth++)
	{
		for (int iH = 0; iH < tensor.size.height; iH++)
		{
			for (int iW = 0; iW < tensor.size.width; iW++)
			{
				os << tensor.values[iH * tensor.dw + iW * tensor.size.depth + iDepth] << " ";
			}
			os << std::endl;
		}

		os << std::endl;
	}

	return os;
}
