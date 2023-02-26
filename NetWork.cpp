#include "stdafx.h"

#include "NetWork.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

using namespace std;

void SoftMax(double* input, size_t size)
{
	const double max = *std::max_element(&input[0], &input[0] + size);

	double sum = 0.0;
	for (size_t i = 0; i < size; ++i) 
	{
		sum += exp(input[i] - max);
	}

	const double constant = max + log(sum);
	for (size_t i = 0; i < size; ++i)
	{
		input[i] = exp(input[i] - constant);
	}
}

void NetWork::Init(const int _amountLayers, const int *_layerSize, const bool _useSoftMax, activateFunction _functionId)
{
	this->useSoftMax = _useSoftMax;
	functionId       = _functionId;
	nLayers          = _amountLayers;

	this->layersSize = new int[nLayers];
	for (int i = 0; i < nLayers; i++)
	{
		layersSize[i] = _layerSize[i];
	}
	
	weights = new Matrix[nLayers - 1];
	bias    = new double*[nLayers - 1];

	for (int i = 0; i < nLayers - 1; i++) 
	{
		weights[i].Init(layersSize[i + 1], layersSize[i]);
		bias[i] = new double[layersSize[i + 1]];
		weights[i].Rand();
		for (int j = 0; j < layersSize[i + 1]; j++) 
		{
			bias[i][j] = 0.01;
		}
	}
	neuronsValues = new double*[nLayers]; neuronsErrors = new double*[nLayers];
	for (int i = 0; i < nLayers; i++) 
	{
		neuronsValues[i] = new double[layersSize[i]]; 
		neuronsErrors[i] = new double[layersSize[i]];
	}
}
void NetWork::SaveWeights() 
{
	ofstream fout;
	fout.open("Weights.txt");
	if (!fout.is_open()) 
	{
		cout << "Error reading the file";
		system("pause");
	}
	for (int i = 0; i < nLayers - 1; ++i)
		fout << weights[i] << " ";

	for (int i = 0; i < nLayers - 1; ++i) 
	{
		for (int j = 0; j < layersSize[i + 1]; ++j) 
		{
			fout << bias[i][j] << " ";
		}
	}
	cout << "Weights saved \n";
	fout.close();
}
void NetWork::SetInput(double* values) 
{
	memcpy(&neuronsValues[0][0], &values[0], layersSize[0] * sizeof(double));
}
double NetWork::ForwardFeed() 
{
	for (int k = 1; k < nLayers; ++k) 
	{
		Matrix::MatMult(weights[k - 1], neuronsValues[k - 1], neuronsValues[k]);
		Matrix::VectorsSum(neuronsValues[k], bias[k - 1], layersSize[k]);
		ActivateNeuronLayer(neuronsValues[k], layersSize[k], functionId);
	}
	SoftMax(&neuronsValues[nLayers-1][0],layersSize[nLayers - 1]);
	return GetArrayMaxPlace(neuronsValues[nLayers - 1]);
}
int NetWork::GetArrayMaxPlace(double* value) 
{
	const int lastLayerIndex = nLayers - 1;
	const double *nv = &neuronsValues[lastLayerIndex][0];
	return std::max_element(&nv[0], &nv[0] + layersSize[nLayers - 1])-&neuronsValues[lastLayerIndex][0];
}
void NetWork::BackPropogation(double _expect) 
{
	const int expect = (int) _expect;
	for (int i = 0; i < layersSize[nLayers - 1]; i++)
	{
		neuronsErrors[nLayers - 1][i] = i != expect ? -neuronsValues[nLayers - 1][i] : 1.0 - neuronsValues[nLayers - 1][i];
	}
	
	for (int k = nLayers - 2; k >= 0; k--) 
	{
		Matrix::MatMultTranspose(weights[k], neuronsErrors[k + 1], neuronsErrors[k]);
	}
}
double NetWork::ErrorCounter() 
{
	double err = 0;
	const int tSize = layersSize[nLayers - 1];
	for (int i = 0; i < tSize; i++) 
	{
		err += fabs(neuronsErrors[nLayers - 1][i]);
	}
	return err;
}
void NetWork::WeightsUpdater(double lr) 
{
	for (int i = 0; i < nLayers - 1; ++i) 
	{
		for (int j = 0; j < layersSize[i + 1]; ++j) 
		{
			for (int k = 0; k < layersSize[i]; ++k) 
			{
				weights[i](j, k) += neuronsErrors[i + 1][j] * ActivateFuncDerivative(neuronsValues[i + 1][j], functionId) * neuronsValues[i][k] * lr;
			}
			bias[i][j] += neuronsErrors[i + 1][j] * ActivateFuncDerivative(neuronsValues[i + 1][j], functionId) * lr;
		}
	}
}
void NetWork::SetActivateFunctionType(activateFunction _functionId)
{
	this->functionId = _functionId;
}
double NetWork::ActivateNeuron(const double neuronValue, const activateFunction funcType)
{
	switch (funcType)
	{
	case MRELU:
		return neuronValue > 0 ? neuronValue : 0.01*neuronValue;
		break;
	case SIGMOID:
		return 1.0 / (1.0 + exp(-neuronValue));
		break;
	default:
		printf("Activate function with ID=%d was not found\n", funcType);
		return 0;
		break;
	}
	return 0;
}
double NetWork::ActivateFuncDerivative(const double neuronValue, const activateFunction funcType)
{
	double temp;
	switch (funcType)
	{
	case MRELU:
		return neuronValue > 0 ? 1.0 : 0.01;
		break;
	case SIGMOID:
		temp = ActivateNeuron(neuronValue, SIGMOID);
		return temp * (1.0 - temp);
		break;
	default:
		printf("Activate function with ID=%d was not found\n", funcType);
		return 0;
		break;
	}
	return 0;
}
void NetWork::ActivateNeuronLayer(double *neuronsLayer, const int layerSize, const activateFunction funcType)
{
	switch (funcType)
	{
	case MRELU:
		for (int i = 0; i < layerSize; i++)
		{
			neuronsLayer[i] = neuronsLayer[i] > 0 ? neuronsLayer[i] : 0.01*neuronsLayer[i];
		}
		break;
	case SIGMOID:
		for (int i = 0; i < layerSize; i++)
		{
			neuronsLayer[i] = 1.0 / (1.0 + exp(-neuronsLayer[i]));
		}
		break;
	default:
		printf("Activate function with ID=%d was not found\n", funcType);
		break;
	}
}
void NetWork::ActivateFuncDerivativeForLayer(double *neuronsLayer, const int layerSize, const activateFunction funcType)
{
	switch (funcType)
	{
	case MRELU:
		for (int i = 0; i < layerSize; i++)
		{
			neuronsLayer[i] = neuronsLayer[i] > 0 ? 1.0 : 0.01;
		}
		break;
	case SIGMOID:
		for (int i = 0; i < layerSize; i++)
		{
			const double sigmoid = ActivateNeuron(neuronsLayer[i], SIGMOID);
			neuronsLayer[i] = sigmoid * (1.0 - sigmoid);
		}
		break;
	default:
		printf("Activate function with ID=%d was not found\n", funcType);
		break;
	}
}
void NetWork::ReadWeights() 
{
	ifstream fin;
	fin.open("Weights.txt");
	if (!fin.is_open()) 
	{
		cout << "Error reading the file";
		system("pause");
	}
	for (int i = 0; i < nLayers - 1; ++i) 
	{
		fin >> weights[i];
	}
	for (int i = 0; i < nLayers - 1; ++i) 
	{
		for (int j = 0; j < layersSize[i + 1]; ++j) 
		{
			fin >> bias[i][j];
		}
	}
	cout << "Weights readed \n";
	fin.close();
}
void NetWork::PrintConfig() 
{
	printf("NeuroNet conf:{");
	for (int i = 0; i < nLayers; i++)
	{
		printf("%d", layersSize[i]);
		if (i != nLayers - 1)
		{
			printf(", ");
		}
	}
	printf("}\n");
}
void NetWork::PrintValues(int n) 
{
	for (int j = 0; j < layersSize[n - 1]; j++) 
	{
		cout << j << " " << neuronsValues[n - 1][j] << endl;
	}
}