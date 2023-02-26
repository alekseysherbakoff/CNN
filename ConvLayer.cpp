#include "stdafx.h"
#include "ConvLayer.h"


// создание свёрточного слоя
// инициализация весов по методу Xavier
ConvLayer::ConvLayer(TensorSize size, int _nFilters, int _filterSize, int _nZeroesFilling, int _convolutionStep) : distribution(0.0, sqrt(2.0 / (_filterSize*_filterSize*size.depth)))
{
	// запоминаем входной размер
	inputSize.width  = size.width;
	inputSize.height = size.height;
	inputSize.depth  = size.depth;

	// вычисляем выходной размер
	outputSize.width  = (size.width  - _filterSize + 2 * _nZeroesFilling) / _convolutionStep + 1;
	outputSize.height = (size.height - _filterSize + 2 * _nZeroesFilling) / _convolutionStep + 1;
	outputSize.depth  = _nFilters;

	this->nZeroesFilling  = _nZeroesFilling; // сохраняем дополнение нулями
	this->convolutionStep = _convolutionStep; // сохраняем шаг свёртки

	this->nFilters    = _nFilters; // сохраняем число фильтров
	this->filterSize  = _filterSize; // сохраняем размер фильтров
	this->filterDepth = size.depth; // сохраняем глубину фильтров

	// добавляем fc тензоров для весов фильтров и их градиентов
	filters      = std::vector<Tensor>(_nFilters, Tensor(_filterSize, _filterSize, filterDepth));
	deltaFilters = std::vector<Tensor>(_nFilters, Tensor(_filterSize, _filterSize, filterDepth));

	// добавляем fc нулей для весов смещения и их градиентов
	biases      = std::vector<double>(_nFilters, 0);
	deltaBiases = std::vector<double>(_nFilters, 0);

	InitWeights(); // инициализируем весовые коэффициенты
}
// инициализация весовых коэффициентов
void ConvLayer::InitWeights()
{
	// проходимся по каждому из фильтров
	for (int iFilter = 0; iFilter < nFilters; iFilter++)
	{
		for (int iFilterSize = 0; iFilterSize < filterSize; iFilterSize++)
		{
			for (int jFilterSize = 0; jFilterSize < filterSize; jFilterSize++)
			{
				for (int iDepth = 0; iDepth < filterDepth; iDepth++)
				{
					filters[iFilter](iDepth, iFilterSize, jFilterSize) = distribution(generator); // генерируем случайное число и записываем его в элемент фильтра
				}
			}
		}

		biases[iFilter] = 0.01; // все смещения устанавливаем в 0.01
	}
}
// прямое распространение
Tensor ConvLayer::Forward(const Tensor &X)
{
	Tensor output(outputSize); // создаём выходной тензор
							   // проходимся по каждому из фильтров
	for (int iFilter = 0; iFilter < nFilters; iFilter++)
	{
		for (int iOutH = 0; iOutH < outputSize.height; iOutH++)
		{
			for (int iOutW = 0; iOutW < outputSize.width; iOutW++)
			{
				double outImageElement = biases[iFilter]; // сразу прибавляем смещение
				// проходимся по фильтрам
				for (int iFilterSize = 0; iFilterSize < filterSize; iFilterSize++)
				{
					for (int jFilterSize = 0; jFilterSize < filterSize; jFilterSize++)
					{
						const int i0 = convolutionStep * iOutH + iFilterSize - nZeroesFilling;
						const int j0 = convolutionStep * iOutW + jFilterSize - nZeroesFilling;

						// поскольку вне границ входного тензора элементы нулевые, то просто игнорируем их
						if (i0 < 0 || i0 >= inputSize.height || j0 < 0 || j0 >= inputSize.width)
							continue;

						// проходимся по всей глубине тензора и считаем сумму
						for (int iDepth = 0; iDepth < filterDepth; iDepth++)
						{
							outImageElement += X(iDepth, i0, j0) * filters[iFilter](iDepth, iFilterSize, jFilterSize);
						}
					}
				}
				/// сразу активируем значение. RELU
				outImageElement = outImageElement >= 0 ? outImageElement : 0.01*outImageElement;
				output(iFilter, iOutH, iOutW) = outImageElement;
			}
		}
	}

	return output; // возвращаем выходной тензор
}
// обратное распространение
Tensor ConvLayer::Backward(const Tensor &dout, const Tensor &X)
{
	TensorSize size; // размер дельт

	// расчитываем размер для дельт
	size.height = convolutionStep * (outputSize.height - 1) + 1;
	size.width  = convolutionStep * (outputSize.width - 1) + 1;
	size.depth  = outputSize.depth;

	Tensor deltas(size); // создаём тензор для дельт

	// расчитываем значения дельт
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

	// расчитываем градиенты весов фильтров и смещений
	for (int iFilter = 0; iFilter < nFilters; iFilter++)
	{
		for (int iInputH = 0; iInputH < size.height; iInputH++)
		{
			for (int iInputW = 0; iInputW < size.width; iInputW++)
			{
				double delta = deltas(iFilter, iInputH, iInputW); // запоминаем значение градиента

				for (int iFilterSize = 0; iFilterSize < filterSize; iFilterSize++)
				{
					for (int jFilterSize = 0; jFilterSize < filterSize; jFilterSize++)
					{
						const int i0 = convolutionStep * iInputH + iFilterSize - nZeroesFilling;
						const int j0 = convolutionStep * iInputW + jFilterSize - nZeroesFilling;

						// игнорируем выходящие за границы элементы
						if (i0 < 0 || i0 >= inputSize.height || j0 < 0 || j0 >= inputSize.width)
							continue;

						// наращиваем градиент фильтра
						for (int iDepth = 0; iDepth < filterDepth; iDepth++)
						{
							deltaFilters[iFilter](iDepth, iFilterSize, jFilterSize) += delta * X(iDepth, i0, j0);
						}

					}
				}
				deltaBiases[iFilter] += delta; // наращиваем градиент смещения
			}
		}
	}

	int pad = filterSize - 1 - nZeroesFilling; // заменяем величину дополнения
	Tensor dX(inputSize); // создаём тензор градиентов по входу

						  // расчитываем значения градиента
	for (int iInputH = 0; iInputH < inputSize.height; iInputH++)
	{
		for (int iInputW = 0; iInputW < inputSize.width; iInputW++)
		{
			for (int iInDepth = 0; iInDepth < filterDepth; iInDepth++)
			{
				double gradientElement = 0; // сумма для градиента

				// идём по всем весовым коэффициентам фильтров
				for (int iFilterSize = 0; iFilterSize < filterSize; iFilterSize++)
				{
					for (int jFilterSize = 0; jFilterSize < filterSize; jFilterSize++)
					{
						const int i0 = iInputH + iFilterSize - pad;
						const int j0 = iInputW + jFilterSize - pad;

						// игнорируем выходящие за границы элементы
						if (i0 < 0 || i0 >= size.height || j0 < 0 || j0 >= size.width)
							continue;

						// суммируем по всем фильтрам
						for (int iFilter = 0; iFilter < nFilters; iFilter++)
						{
							gradientElement += filters[iFilter](iInDepth, filterSize - 1 - iFilterSize, filterSize - 1 - jFilterSize) * deltas(iFilter, i0, j0); // добавляем произведение повёрнутых фильтров на дельты
						}
					}
				}

				dX(iInDepth, iInputH, iInputW) = gradientElement; // записываем результат в тензор градиента
			}
		}
	}

	return dX; // возвращаем тензор градиентов
}
// обновление весовых коэффициентов
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
					filters[iFilter](iDepth, iFilterSize, jFilterSize) -= learningRate * deltaFilters[iFilter](iDepth, iFilterSize, jFilterSize); // вычитаем градиент, умноженный на скорость обучения
					deltaFilters[iFilter](iDepth, iFilterSize, jFilterSize) = 0; // обнуляем градиент фильтра
				}
			}
		}

		biases[iFilter] -= learningRate * deltaBiases[iFilter]; // вычитаем градиент, умноженный на скорость обучения
		deltaBiases[iFilter] = 0; // обнуляем градиент веса смещения
	}
}
TensorSize ConvLayer::GetOutputSize()
{
	return this->outputSize;
}
