#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include "../inc/camera_calibrate_functions.h"

using namespace cv;
using namespace std;

// ������� ��� �������� ������ 3D-����� ����� ��������� �����
void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners) {
    for (int i = 0; i < boardSize.height; i++) { // ������� �� ������ �����
        for (int j = 0; j < boardSize.width; j++) { // ������� �� ������ �����
            // ���������� ����� � ������
            corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
        }
    }
}

// ������� ��� ���������� ����� ��������� ����� �� ������ �����������
void getChessBoardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults) {
    for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++) { // ������� ���� �����������
        vector<Point2f> pointBuf;
        // ����� ����� ��������� �����
        bool found = findChessboardCorners(*iter, Size(9, 6), pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        if (found) { // ���� ���� �������
            allFoundCorners.push_back(pointBuf); // ��������� �� � ����� ������
        }

        if (showResults) { // ���� ���������� ���������� ����������
            drawChessboardCorners(*iter, Size(9, 6), pointBuf, found); // ��������� �����
            imshow("Looking for corners", *iter); // ����� �����������
            waitKey(0); // �������� ������� �������
        }
    }
}

// ������� ��� ���������� ������
void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients) {
    vector<vector<Point2f>> checkerboardImageSpacePoints;
    // ��������� ����� ��������� ����� �� ���� ������������
    getChessBoardCorners(calibrationImages, checkerboardImageSpacePoints,true);

    vector<vector<Point3f>> worldSpaceCornerPoints(1);
    // �������� 3D-����� ����� ��������� �����
    createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
    worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

    vector<Mat> rVectors, tVectors; // ������� ��� �������� ����������� �������� � ����������
    distanceCoefficients = Mat::zeros(8, 1, CV_64F); // ������������� ������������� ���������

    // ���������� ������
    calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);
}

// ������� ��� ���������� ���������� ������ � ����
bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients) {
    ofstream outStream(name);
    if (outStream) {
        uint16_t rows = cameraMatrix.rows; // ���������� ����� � ������� ������
        uint16_t columns = cameraMatrix.cols; // ���������� �������� � ������� ������

        outStream << rows << endl;
        outStream << columns << endl;

        // ���������� ������� ������
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                double value = cameraMatrix.at<double>(r, c);
                outStream << value << endl;
            }
        }

        rows = distanceCoefficients.rows; // ���������� ����� � ������� ������������� ���������
        columns = distanceCoefficients.cols; // ���������� �������� � ������� ������������� ���������

        outStream << rows << endl;
        outStream << columns << endl;

        // ���������� ������������� ���������
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                double value = distanceCoefficients.at<double>(r, c);
                outStream << value << endl;
            }
        }
        outStream.close();
        return true;
    }
    return false;
}

// ������� ��� �������� ���������� ������ �� �����
bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients) {
    ifstream instream(name);
    if (instream) {
        uint16_t rows;
        uint16_t columns;

        // �������� ������� ������
        instream >> rows;
        instream >> columns;
        cameraMatrix = Mat(Size(columns, rows), CV_64F);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                double read = 0.0f;
                instream >> read;
                cameraMatrix.at<double>(r, c) = read;
                std::cout << cameraMatrix.at<double>(r, c) << "\n";
            }
        }

        // �������� ������������� ���������
        instream >> rows;
        instream >> columns;
        distanceCoefficients = Mat::zeros(rows, columns, CV_64F);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                double read = 0.0f;
                instream >> read;
                distanceCoefficients.at<double>(r, c) = read;
                std::cout << distanceCoefficients.at<double>(r, c) << "\n";
            }
        }
        instream.close();
        return true;
    }
    return false;
}

// �������� �������, ������� ��������� ���������� ������ ��� ��������� ��������� �� �����
void calibrateOrLoadCamera(bool recalibrate, string calibrationFile, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients) {


    if (recalibrate) {
        // ������ ��� �������� ����������� ��� ����������
        vector<Mat> calibrationImages;

        // ����������� � ���-������
        VideoCapture vid(0);
        if (!vid.isOpened()) {
            cerr << "Error: Could not open video stream." << endl;
            return;
        }

        // ���� ��� ����������� �����������
        namedWindow("Webcam", WINDOW_AUTOSIZE);

        int imagesCaptured = 0;
        while (imagesCaptured < 20) {
            Mat frame;
            bool success = vid.read(frame);
            if (!success) {
                cerr << "Error: Could not read frame from video stream." << endl;
                break;
            }

            imshow("Webcam", frame);
            char key = waitKey(30);

            if (key == ' ') { // ������� ������� ��� ������� �����������
                calibrationImages.push_back(frame.clone());
                imagesCaptured++;
                cout << "Captured image " << imagesCaptured << "/20" << endl;
            }
        }

        // �������� ���� � ������������ �����������
        cv::destroyWindow("Webcam");
        vid.release();

        // ��������, ��� ���� ��������� ���������� ����������� ��� ����������
        if (calibrationImages.size() < 20) {
            cerr << "Error: Not enough images captured for calibration." << endl;
            return;
        }

        // ���������� ������
        cameraCalibration(calibrationImages, boardSize, squareEdgeLength, cameraMatrix, distanceCoefficients);

        // ���������� ����������� ����������
        saveCameraCalibration(calibrationFile, cameraMatrix, distanceCoefficients);
    }
    else {
        // �������� ����������� ����������
        if (!loadCameraCalibration(calibrationFile, cameraMatrix, distanceCoefficients)) {
            cerr << "Error: Could not load calibration data from file." << endl;
            return;
        }
    }
}