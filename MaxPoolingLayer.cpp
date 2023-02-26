#include "stdafx.h"
#include "MaxPoolingLayer.h"
#include "Tensor.h"

// создание слоя
MaxPoolingLayer::MaxPoolingLayer(TensorSize size, int _scale) : mask(size) 
{
	// запоминаем входной размер
	inputSize.width = size.width;
	inputSize.height = size.height;
	inputSize.depth = size.depth;

	// вычисляем выходной размер
	outputSize.width = size.width / _scale;
	outputSize.height = size.height / _scale;
	outputSize.depth = size.depth;

	this->scale = _scale; // запоминаем коэффициент уменьшения
}

// прямое распространение с использованием маски
Tensor MaxPoolingLayer::Forward(const Tensor &X) 
{
	Tensor output(outputSize); // создаём выходной тензор

	// проходимся по каждому из каналов
	for (int iDepth = 0; iDepth < inputSize.depth; iDepth++) 
	{
		for (int iInputH = 0; iInputH < inputSize.height; iInputH += scale) 
		{
			for (int iInputW = 0; iInputW < inputSize.width; iInputW += scale) 
			{
				int imax = iInputH; // индекс строки максимума
				int jmax = iInputW; // индекс столбца максимума
				double max = X(iDepth, iInputH, iInputW); // начальное значение максимума - значение первой клетки подматрицы
				const int i0 = iInputH / scale;
				const int j0 = iInputW / scale;
				if (i0 < 0 || i0 >= outputSize.height || j0 < 0 || j0 >= outputSize.width)
					continue;
				// проходимся по подматрице и ищем максимум и его координаты
				for (int y = iInputH; y < iInputH + scale; y++) 
				{
					for (int x = iInputW; x < iInputW + scale; x++) 
					{
						double value = X(iDepth, y, x); // получаем значение входного тензора
						mask(iDepth, y, x) = 0; // обнуляем маску

						// если входное значение больше максимального
						if (value > max) 
						{
							max = value; // обновляем максимум
							imax = y; // обновляем индекс строки максимума
							jmax = x; // обновляем индекс столбца максимума
						}
					}
				}
				output(iDepth, i0, j0) = max; // записываем в выходной тензор найденный максимум
				mask(iDepth, imax, jmax) = 1; // записываем 1 в маску в месте расположения максимального элемента
			}
		}
	}

	return output; // возвращаем выходной тензор
}

// обратное распространение
Tensor MaxPoolingLayer::Backward(const Tensor &dout, const Tensor &X) 
{
	Tensor dX(inputSize); // создаём тензор для градиентов
	for (int iDepth = 0; iDepth < inputSize.depth; iDepth++)
	{
		for (int iInputH = 0; iInputH < inputSize.height; iInputH++)
		{
			for (int iInputW = 0; iInputW < inputSize.width; iInputW++)
			{
				dX(iDepth, iInputH, iInputW) = dout(iDepth, iInputH / scale, iInputW / scale) * mask(iDepth, iInputH, iInputW); // умножаем градиенты на маску
			}
		}
	}

	return dX; // возвращаем посчитанные градиенты
}