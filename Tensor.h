#pragma once

#include <vector>
#include <iostream>

using namespace std;

// размерность тензора
struct TensorSize
{
	int depth; // глубина
	int height; // высота
	int width; // ширина
	TensorSize() {};
	TensorSize(int d, int h, int w) : depth(d), height(h), width(w) {};
};

// тензор
class Tensor
{
public:
	/// Создание тензора
	void Init(int width, int height, int depth);
	/// Передача элементов тензора
	void SetValues(double *val);
	/// Вовзрат элементов тензора
	void GetValues(double *val);
	/// Создание тензора по размерности
	Tensor(int width, int height, int depth);
	/// Создание тензора по размерности
	Tensor(const TensorSize &size); // создание из размера
	/// Перегруженный оператор индексации
	double& operator()(int iDepth, int iHeight, int iWidth);
	/// Перегруженный оператор индексации
	double operator()(int iDepth, int iHeight, int iWidth) const;
	/// Сложение двух тензоров
	Tensor operator+(const  Tensor &other);
	/// Разность двух тензоров
	Tensor operator-(const  Tensor &other);
	/// Умножение элементов тензора на число
	Tensor operator*(const double alpha);
	/// Получение размеров тензора
	TensorSize GetSize() const;
	/// Печать элементов тензора в консоль
	friend std::ostream& operator<<(std::ostream& os, const Tensor &tensor);

private:
	/// Размерность тензора
	TensorSize size;
	/// Значения элементов тензора
	std::vector<double> values;
	/// параметр нужный для индексации. т.к. трехмерный массив упакован в одномерный
	int dw;

};
