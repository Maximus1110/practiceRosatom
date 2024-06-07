#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;

// ������� ��� �������� ������ 3D-����� ����� ��������� �����
void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners);

// ������� ��� ���������� ����� ��������� ����� �� ������ �����������
void getChessBoardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults);

// ������� ��� ���������� ������
void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients);

// ������� ��� ���������� ���������� ������ � ����
bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients);

// ������� ��� �������� ���������� ������ �� �����
bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients);

// �������� ������� ��� ������ ����������
void calibrateOrLoadCamera(bool recalibrate, string calibrationFile, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients);