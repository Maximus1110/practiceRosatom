#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include "../inc/camera_calibrate_functions.h"

using namespace cv;
using namespace std;

// Функция для создания списка 3D-точек углов шахматной доски
void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners) {
    for (int i = 0; i < boardSize.height; i++) { // Перебор по высоте доски
        for (int j = 0; j < boardSize.width; j++) { // Перебор по ширине доски
            // Добавление углов в вектор
            corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
        }
    }
}

// Функция для нахождения углов шахматной доски на наборе изображений
void getChessBoardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults) {
    for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++) { // Перебор всех изображений
        vector<Point2f> pointBuf;
        // Поиск углов шахматной доски
        bool found = findChessboardCorners(*iter, Size(9, 6), pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        if (found) { // Если углы найдены
            allFoundCorners.push_back(pointBuf); // Добавляем их в общий вектор
        }

        if (showResults) { // Если необходимо показывать результаты
            drawChessboardCorners(*iter, Size(9, 6), pointBuf, found); // Отрисовка углов
            imshow("Looking for corners", *iter); // Показ изображения
            waitKey(0); // Ожидание нажатия клавиши
        }
    }
}

// Функция для калибровки камеры
void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients) {
    vector<vector<Point2f>> checkerboardImageSpacePoints;
    // Получение углов шахматной доски на всех изображениях
    getChessBoardCorners(calibrationImages, checkerboardImageSpacePoints,true);

    vector<vector<Point3f>> worldSpaceCornerPoints(1);
    // Создание 3D-точек углов шахматной доски
    createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
    worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

    vector<Mat> rVectors, tVectors; // Векторы для хранения результатов вращения и трансляции
    distanceCoefficients = Mat::zeros(8, 1, CV_64F); // Инициализация коэффициентов дисторсии

    // Калибровка камеры
    calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);
}

// Функция для сохранения параметров камеры в файл
bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients) {
    ofstream outStream(name);
    if (outStream) {
        uint16_t rows = cameraMatrix.rows; // Количество строк в матрице камеры
        uint16_t columns = cameraMatrix.cols; // Количество столбцов в матрице камеры

        outStream << rows << endl;
        outStream << columns << endl;

        // Сохранение матрицы камеры
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                double value = cameraMatrix.at<double>(r, c);
                outStream << value << endl;
            }
        }

        rows = distanceCoefficients.rows; // Количество строк в матрице коэффициентов дисторсии
        columns = distanceCoefficients.cols; // Количество столбцов в матрице коэффициентов дисторсии

        outStream << rows << endl;
        outStream << columns << endl;

        // Сохранение коэффициентов дисторсии
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

// Функция для загрузки параметров камеры из файла
bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients) {
    ifstream instream(name);
    if (instream) {
        uint16_t rows;
        uint16_t columns;

        // Загрузка матрицы камеры
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

        // Загрузка коэффициентов дисторсии
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

// Основная функция, которая выполняет калибровку камеры или загружает параметры из файла
void calibrateOrLoadCamera(bool recalibrate, string calibrationFile, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients) {


    if (recalibrate) {
        // Вектор для хранения изображений для калибровки
        vector<Mat> calibrationImages;

        // Подключение к веб-камере
        VideoCapture vid(0);
        if (!vid.isOpened()) {
            cerr << "Error: Could not open video stream." << endl;
            return;
        }

        // Окно для отображения видеопотока
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

            if (key == ' ') { // Нажатие пробела для захвата изображения
                calibrationImages.push_back(frame.clone());
                imagesCaptured++;
                cout << "Captured image " << imagesCaptured << "/20" << endl;
            }
        }

        // Закрытие окна и освобождение видеопотока
        cv::destroyWindow("Webcam");
        vid.release();

        // Проверка, что было захвачено достаточно изображений для калибровки
        if (calibrationImages.size() < 20) {
            cerr << "Error: Not enough images captured for calibration." << endl;
            return;
        }

        // Калибровка камеры
        cameraCalibration(calibrationImages, boardSize, squareEdgeLength, cameraMatrix, distanceCoefficients);

        // Сохранение результатов калибровки
        saveCameraCalibration(calibrationFile, cameraMatrix, distanceCoefficients);
    }
    else {
        // Загрузка результатов калибровки
        if (!loadCameraCalibration(calibrationFile, cameraMatrix, distanceCoefficients)) {
            cerr << "Error: Could not load calibration data from file." << endl;
            return;
        }
    }
}