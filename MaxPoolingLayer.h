#pragma once

#include "Tensor.h"
class MaxPoolingLayer 
{
public:
	MaxPoolingLayer(TensorSize size, int scale = 2); // создание слоя
	Tensor Forward(const Tensor &X); // прямое распространение
	Tensor Backward(const Tensor &dout, const Tensor &X); // обратное распространение
	TensorSize GetInputSize() { return inputSize; };
	TensorSize GetOutputSize() { return outputSize; };

private:
	TensorSize inputSize; // размер входа
	TensorSize outputSize; // размер выхода

	int scale; // во сколько раз уменьшается размерность
	Tensor mask; // маска для максимумов
};