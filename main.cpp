#include "stdafx.h"

#include "NetWork.h"
#include "Tensor.h"
#include "ConvLayer.h"
#include "MaxPoolingLayer.h"

unsigned int in(ifstream& icin, unsigned int size) 
{
	unsigned int ans = 0;
	for (int i = 0; i < size; i++) 
	{
		unsigned char x;
		icin.read((char*)&x, 1);
		unsigned int temp = x;
		ans <<= 8;
		ans += temp;
	}
	return ans;
}

void readMnist(double ***trainData, double **trainAnswers, double ***verData, double **verAnswers, int &nTrainExamples, int &nVerExamples)
{
	printf("***MNIST reading:\n");
	unsigned int num, magic, rows, cols;
	string dirPath = "C:/Users/Алексей/Desktop/Work/NeuroNet/mnist-master/";
	string MNIST_FileNames[4] = { "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", "train-images.idx3-ubyte", "train-labels.idx1-ubyte" };
	ifstream file;
	string fileName = dirPath + MNIST_FileNames[0];

	ifstream icin;
	icin.open(dirPath + "train-images.idx3-ubyte", ios::binary);

	magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);
	nTrainExamples = num;

	(*trainData) = new double *[num];
	(*trainAnswers) = new double[num];
	printf("\tReading train data...\n");
	for (int i = 0; i < num; i++)
	{
		(*trainData)[i] = new double[rows*cols];
		for (int x = 0; x < rows; x++)
		{
			for (int y = 0; y < cols; y++)
			{
				int temp = in(icin, 1);
				(*trainData)[i][x*cols + y] = (double)temp / 255.0;
			}
		}
	}
	icin.close();
	printf("\tReading train answers...\n");
	icin.open(dirPath + "train-labels.idx1-ubyte", ios::binary);
	magic = in(icin, 4), num = in(icin, 4);
	for (int i = 0; i < num; i++)
	{
		(*trainAnswers)[i] = in(icin, 1);
	}
	icin.close();
	printf("\tReading ver data...\n");
	icin.open(dirPath + "t10k-images.idx3-ubyte", ios::binary);
	magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);

	nVerExamples = num;

	(*verData) = new double *[num];
	(*verAnswers) = new double[num];

	for (int i = 0; i < num; i++)
	{
		(*verData)[i] = new double[rows*cols];
		for (int x = 0; x < rows; x++)
		{
			for (int y = 0; y < cols; y++)
			{
				int temp = in(icin, 1);
				(*verData)[i][x*cols + y] = (double)temp / 255.0;
			}
		}
	}
	icin.close();
	printf("\tReading ver answers...\n");
	icin.open(dirPath + "t10k-labels.idx1-ubyte", ios::binary);
	magic = in(icin, 4), num = in(icin, 4);
	for (int i = 0; i < num; i++)
	{
		(*verAnswers)[i] = in(icin, 1);
	}
	icin.close();

	printf("\tMNIST was sucessfull read\n\n");
}

int main()
{
	double **trainData = nullptr, *trainAnswers = nullptr, **verData = nullptr, *verAnswers = nullptr;
	int nTrainExamples, nVerExamples;

	readMnist(&trainData, &trainAnswers, &verData, &verAnswers, nTrainExamples, nVerExamples);

	printf("***Education Data:\n\tAmount train examples = %d\n\tAmount ver examples   = %d\n\n", nTrainExamples, nVerExamples);
#define CNN
#ifdef CNN
	const int filterSize      = 3;
	const int convolutionStep = 1;
	const int nZeroesFilling  = 0;
	const int nFilters        = 1; // глубина ядра
	const int imageWidth      = 28; // ширина входного изображения
	const int imageHeight     = 28; // высота входного иозображения
	const int nEpochs         = 10; 
	const int poolingReduceImage = 2;
	const double fullConnectedNNLearningRay = 1e-3;
	const double convolutionalNNLearningRay = 2e-3;

	TensorSize tSize(nFilters, imageWidth, imageHeight);
	ConvLayer convLayer(tSize, nFilters, filterSize, nZeroesFilling, convolutionStep);
	TensorSize outputCLayerSize = convLayer.GetOutputSize();

	MaxPoolingLayer poolingLayer(outputCLayerSize, poolingReduceImage);
	TensorSize outputPoolingLayerSize = poolingLayer.GetOutputSize();

	const int firstFullConnectedNNSize = outputPoolingLayerSize.depth*outputPoolingLayerSize.width*outputPoolingLayerSize.height;
	printf("ConvLayer:    {filterSize=%d, convStep=%d, nZeroesFilling=%d, nFilters=%d}\nPoolingLayer: {poolingReduceImage=%d}\n",
			filterSize, convolutionStep, nZeroesFilling, nFilters, poolingReduceImage);

#else
	const double fullConnectedNNLearningRay = 1e-3;
	const int firstFullConnectedNNSize      = 784;
	const int nEpochs						= 10;
#endif
	const bool useSoftMax = true;
	const int nLayers= 3;
	const int layersSize[3] = { firstFullConnectedNNSize, 30, 10 };
	NetWork nn;

	nn.Init(nLayers, layersSize, useSoftMax, MRELU);
	nn.PrintConfig();
	printf("\n***Education:\n");
	double *buffer = new double[layersSize[0]];
	for (int iEpoch = 0; iEpoch < nEpochs; iEpoch++)
	{
		int accuracy = 0;
		double loss = 0.0;	

		for (int iExample = 0; iExample < nTrainExamples; iExample++)
		{
#ifdef CNN
			Tensor inputTensor(imageWidth, imageHeight, nFilters);
			inputTensor.SetValues(&trainData[iExample][0]);

			Tensor convOutput    = convLayer.Forward(inputTensor);	
			Tensor poolingOutput = poolingLayer.Forward(convOutput);
			poolingOutput.GetValues(&buffer[0]);

			nn.SetInput(&buffer[0]);
#else
			nn.SetInput(&trainData[iExample][0]);
#endif
			accuracy = nn.ForwardFeed() == (int)trainAnswers[iExample] ? accuracy + 1 : accuracy;

			nn.BackPropogation(trainAnswers[iExample]);
			nn.WeightsUpdater(fullConnectedNNLearningRay);
			loss += fabs(nn.ErrorCounter()) / (double)nTrainExamples;
#ifdef CNN
			TensorSize poolingOutputSize = poolingLayer.GetOutputSize();
			Tensor dout(poolingOutputSize);
			dout.SetValues(nn.neuronsErrors[0]);

			Tensor poolingBackward = poolingLayer.Backward(dout, poolingOutput);
			Tensor convLayerError  = convLayer.Backward(poolingBackward, convOutput);
			convLayer.UpdateWeights(convolutionalNNLearningRay);
#endif
			if (iExample % 200 == 0)
				printf("\r\tExample=%d\tloss=%2.3e\taccuracy=%f%%", iExample, loss, (double)accuracy / (double)nTrainExamples * 100);
		}
		printf("\r\t\t\t\t\t\t\t\t\r");
		const double correctAnswersPercent = (double)accuracy / (double)nTrainExamples * 100;
		printf("\tEpoch=%d:\n", iEpoch);
		printf("\t\tTrain: Accuracy = %f%% Loss = %2.3e\n", correctAnswersPercent, loss);
		accuracy = 0;
		loss = 0.0;

		for (int iVerExample = 0; iVerExample < nVerExamples; iVerExample++)
		{
#ifdef CNN
			Tensor inputTensor(imageWidth, imageHeight, nFilters);
			inputTensor.SetValues(&verData[iVerExample][0]);

			Tensor convOutput = convLayer.Forward(inputTensor);
			Tensor poolingOutput = poolingLayer.Forward(convOutput);
			poolingOutput.GetValues(&buffer[0]);

			nn.SetInput(&buffer[0]);
#else
			nn.SetInput(&verData[iVerExample][0]);
#endif
			accuracy = nn.ForwardFeed() == (int)verAnswers[iVerExample] ? accuracy + 1 : accuracy;
			nn.BackPropogation(verAnswers[iVerExample]);
			loss += fabs(nn.ErrorCounter()) / (double)nVerExamples;
		}
		const double correctVerAnswersPercent = (double)accuracy / (double)nVerExamples * 100;
		printf("\t\tVerif: Accuracy = %f%% Loss = %2.3e\n", correctVerAnswersPercent, loss);
	
	}
	delete[] buffer;

	delete[] trainData;
	delete[] trainAnswers;
	delete[] verData;
	delete[] verAnswers;

	return 0;
}