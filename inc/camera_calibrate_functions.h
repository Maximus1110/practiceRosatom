#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;

// Функция для создания списка 3D-точек углов шахматной доски
void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners);

// Функция для нахождения углов шахматной доски на наборе изображений
void getChessBoardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults);

// Функция для калибровки камеры
void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients);

// Функция для сохранения параметров камеры в файл
bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients);

// Функция для загрузки параметров камеры из файла
bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients);

// Основная функция для вызова калибровки
void calibrateOrLoadCamera(bool recalibrate, string calibrationFile, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients);