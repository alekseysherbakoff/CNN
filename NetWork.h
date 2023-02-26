#pragma once

#include "stdafx.h"

#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include <chrono>

#include "Matrix.h"

enum activateFunction
{
	MRELU, SIGMOID
};

class NetWork
{
public:
	activateFunction functionId;
	bool useSoftMax;
	int nLayers;
	int* layersSize;
	double** bias;
	double** neuronsValues, **neuronsErrors;
	Matrix* weights;
public:
	void SetActivateFunctionType(activateFunction _functionId);
	void Init(const int _amountLayers, const int *_layerSize, const bool _useSoftMax = false, activateFunction _functionId = MRELU);
	void SaveWeights();
	void ReadWeights();
	void SetInput(double* values);
	void BackPropogation(double expect);
	void WeightsUpdater(double lr);
	void PrintConfig();
	void PrintValues(int n);

	double ForwardFeed();
	double ErrorCounter();

private:
	int    GetArrayMaxPlace(double* value);
	double ActivateNeuron(const double neuronValue, const activateFunction funcType);
	double ActivateFuncDerivative(const double neuronValue, const activateFunction funcType);
	void   ActivateNeuronLayer(double *neuronsLayer, const int layerSize, const activateFunction funcType);
	void   ActivateFuncDerivativeForLayer(double *neuronsLayer, const int layerSize, const activateFunction funcType);

};