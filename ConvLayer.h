#pragma once
#include "Tensor.h"

#include <random>

using namespace std;

class ConvLayer
{
public:
	/// Создание сверточного слоя
	ConvLayer(TensorSize size, int _nFilters, int _filterSize, int _nZeroesFilling, int _convolutionStep);
	/// Получение размера выхода
	TensorSize GetOutputSize();
	/// Применение фильтра к изображению. Прямой ход
	Tensor Forward(const Tensor &X);
	/// Обратный ход. Вычисление коррекций к фильтру. Возвращает тензор ошибки для предыдущих слоев пулинга и свертки
	Tensor Backward(const Tensor &dout, const Tensor &X);
	/// Обновление фильтра
	void UpdateWeights(const double learningRate);

private:
	///Инициализация фильтра
	void InitWeights(); 

private:
	/// Генератор случайных чисел
	std::default_random_engine generator;
	/// Генерация случайных чисел по нормальному распределению Гаусса
	std::normal_distribution<double> distribution;
	///Размер входа сверточного слоя 
	TensorSize inputSize;
	/// Размер выхода сверточного слоя
	TensorSize outputSize;
	/// Фильтр
	std::vector<Tensor> filters;
	/// Смещения
	std::vector<double> biases;
	/// Величины коррекции фильтров
	std::vector<Tensor> deltaFilters;
	/// Величины коррекции смещений
	std::vector<double> deltaBiases;
	/// Дополнение нулями
	int nZeroesFilling;
	/// Шаг свертки
	int convolutionStep;
	/// Количество фильтров. Для Mnist 1. т.к. изображение не в RGB, а в сером цвете
	int nFilters;
	/// Размер фильтра
	int filterSize;
	/// Глубина фильтров
	int filterDepth;
};