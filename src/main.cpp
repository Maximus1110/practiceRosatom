#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/calib3d.hpp>
#include<iomanip>
#include<sstream>
#include <iostream>
#include <fstream>
#include<ctime>
#include <math.h>
#include <Eigen>
#include "../inc/camera_calibrate_functions.h"

using namespace std;
using namespace cv;
using namespace Eigen;

struct Parameters {
    Size boardSize;
    double squareEdgeLength;
    double arucoSquareDimension;
    Mat cameraMatrix;
    Mat distanceCoefficients;
};

Mat getBinaryImage(const Mat& input, int border = 128) {  
    Mat bin;
    bin.create(input.size(), input.type());

    for (int x = 0; x < input.rows; x++) {
        for (int y = 0; y < input.cols; y++) {
            if (input.at<uchar>(x, y) > border) {
                bin.at<uchar>(x, y) = 255;
            }
            else {
                bin.at<uchar>(x, y) = 0;
            }
        }
    }
    return bin;
}

Mat Sobel(const Mat& input) {
    CV_Assert(input.type() == CV_8UC1);

    Mat bin = Mat::zeros(input.size(), input.type());

    // Оператор Собеля
    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int x = 1; x < input.rows - 1; x++) {
        for (int y = 1; y < input.cols - 1; y++) {
            int buffX = 0;
            int buffY = 0;


            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    buffX += input.at<uchar>(x + i, y + j) * Gx[i + 1][j + 1];
                    buffY += input.at<uchar>(x + i, y + j) * Gy[i + 1][j + 1];
                }
            }


            int magnitude = sqrt(buffX * buffX + buffY * buffY);
            magnitude = min(255,max(0, magnitude));

            bin.at<uchar>(x, y) = static_cast<uchar>(magnitude);
        }
    }

    return bin;
}

Mat Border(const Mat& input) {
    CV_Assert(input.type() == CV_8UC1);

    Mat bin = Mat::zeros(input.size(), input.type());
    bin.setTo(Scalar(255));  // Заполнение белым цветом

    for (int y = 3; y < input.rows - 3; y++) {
        for (int x = 3; x < input.cols - 3; x++) {
            if (input.at<uchar>(y, x - 1) == 255 && input.at<uchar>(y, x) == 0) {
                bin.at<uchar>(y, x) = 0;
            }
            if (input.at<uchar>(y, x) == 0 && input.at<uchar>(y, x + 1) == 255) {
                bin.at<uchar>(y, x) = 0;
            }
            if (input.at<uchar>(y - 1, x) == 255 && input.at<uchar>(y, x) == 0) {
                bin.at<uchar>(y, x) = 0;
            }
            if (input.at<uchar>(y, x) == 0 && input.at<uchar>(y + 1, x) == 255) {
                bin.at<uchar>(y, x) = 0;
            }
        }

    }
    return bin;
}

vector<Point> angleFast(const Mat& image) {
    // Предполагается, что входное изображение - бинарное
    Mat bin;
    if (image.channels() > 1) {
        cvtColor(image, bin, COLOR_BGR2GRAY);
        threshold(bin, bin, 128, 255, THRESH_BINARY);
    }
    else {
        bin = image.clone();
    }

    std::vector<Point> points;


    /*int G[7][7] = {
        {0, 0, 1, 1, 1, 0, 0},
        {0, 1, 0, 0, 0, 1, 0},
        {1, 0, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 1},
        {0, 1, 0, 0, 0, 1, 0},
        {0, 0, 1, 1, 1, 0, 0},
    };*/

    int G[5][5] = {
        {0, 0, 1, 0, 0},
        {0, 1, 0, 1, 0},
        {1, 0, 0, 0, 1},
        {0, 1, 0, 1, 0},
        {0, 0, 1, 0, 0},
    };

    for (int y = 5; y < bin.rows - 5; y++) {
        for (int x = 5; x < bin.cols - 5; x++) {
            if (bin.at<uchar>(y, x) == 0) {
                int kol = 0;
                for (int i = -2; i <= 2; i++) {
                    for (int j = -2; j <= 2; j++) {
                        kol += bin.at<uchar>(y + i, x + j) * G[i + 2][j + 2];
                    }
                }
                if (kol / 255 > 3) {
                    points.push_back(Point(x, y));
                }
            }
        }
    }

    return points;
}

vector<vector<Point>> Sectors(const Mat& input) {
    CV_Assert(input.type() == CV_8UC1);
    int num = 0;

    vector<vector<Point>> sectors_vector = vector<vector<Point>>();
    vector<Point> sector = vector<Point>();

    Mat bin = input.clone();


    int numOfSector = 2;


    for (int y = 2; y < bin.rows - 2; y++) {
        for (int x = 2; x < bin.cols - 2; x++) {


            if (bin.at<uchar>(y, x) == 0) {


                bin.at<uchar>(y, x) = numOfSector;
                sector.push_back(Point(x, y));

                bool isCommonSector = true;

                int x_buf = x;
                int y_buf = y;

                while (isCommonSector && numOfSector < 255)
                {

                    int i_next = 0;
                    int j_next = 0;

                    isCommonSector = false;

                    for (int i = -1; i <= 1; i++) {
                        for (int j = -1; j <= 1; j++) {
                            if (bin.at<uchar>(y_buf + i, x_buf + j) == 0) {
                                j_next = j;
                                i_next = i;
                                bin.at<uchar>(y_buf + i, x_buf + j) = numOfSector;
                                sector.push_back(Point(x_buf + j, y_buf + i));
                                isCommonSector = true;
                            }
                        }
                    }

                    if (!isCommonSector) {
                        for (int i = -2; i <= 2; i++) {
                            for (int j = -2; j <= 2; j++) {
                                if (bin.at<uchar>(y_buf + i, x_buf + j) == 0) {
                                    j_next = j;
                                    i_next = i;
                                    bin.at<uchar>(y_buf + i, x_buf + j) = numOfSector;
                                    sector.push_back(Point(x_buf + j, y_buf + i));
                                    isCommonSector = true;
                                }
                            }
                        }
                    }
                    if (!isCommonSector) {
                        for (int i = -3; i <= 3; i++) {
                            for (int j = -3; j <= 3; j++) {
                                if (bin.at<uchar>(y_buf + i, x_buf + j) == 0) {
                                    j_next = j;
                                    i_next = i;
                                    bin.at<uchar>(y_buf + i, x_buf + j) = numOfSector;
                                    sector.push_back(Point(x_buf + j, y_buf + i));
                                    isCommonSector = true;
                                }
                            }
                        }
                    }

                    y_buf = y_buf + i_next;
                    x_buf = x_buf + j_next;

                    //string Path = "..\\..\\..\\data\\out" + to_string(num) + ".jpg";

                    //imwrite(Path, bin);
                    //num++;
                }

                numOfSector += 1;

                sectors_vector.push_back(sector);
                sector = vector<Point>();

                x = 2;
                y = 2;


            }
        }

    }

    return sectors_vector;
}

double findDistance(Point mainVec, Point pointVec) {
    double distance = 0;


    int scalarProduct = mainVec.x * pointVec.x + mainVec.y * pointVec.y;
    double squareMainVecNorm = pow(mainVec.x, 2) + pow(mainVec.y, 2);


    Point perpendicularVec = Point2d(
        pointVec.x - (scalarProduct * mainVec.x) / squareMainVecNorm,
        pointVec.y - (scalarProduct * mainVec.y) / squareMainVecNorm);

    distance = sqrt(pow(perpendicularVec.x, 2) + pow(perpendicularVec.y, 2));
    return distance;

}

pair<Point,double>  findMaxPerpendicularPoint(const Point& start, const Point& stop, const std::vector<Point>& sector)
{
    Point maxPoint;
    Point mainVec = Point(stop.x - start.x, stop.y - start.y);
    double max_distance = 0;

    for (const auto& p : sector) {


        Point pointVec = Point(p.x - start.x, p.y - start.y);
        double distance = findDistance(mainVec, pointVec);

        if (distance > max_distance) {
            max_distance = distance;
            maxPoint = Point(p);
        }
    }
    return std::make_pair(maxPoint, max_distance);
}

Point findMaxDistancePoint(const Point& first, const std::vector<Point>& sec) {
    Point max_point = sec[0];
    double max_distance = 0;

    for (const auto& p : sec) {
        double dx = p.x - first.x;
        double dy = p.y - first.y;
        double distance = std::sqrt(std::pow(dx, 2) + std::pow(dy, 2));

        if (distance > max_distance) {
            max_distance = distance;
            max_point = p;
        }
    }

    return max_point;
}

bool isSquare(const Point& start, const Point& mean, const Point& stop, const std::vector<Point>& sector, double eps) {
    
    Point mainVec1 = Point(mean.x - start.x, mean.y - start.y);
    Point mainVec2 = Point(stop.x - mean.x, stop.y - mean.y);
    double max_distance = 0;

    for (const auto& p : sector) {


        Point pointVec1 = Point(p.x - start.x, p.y - start.y);
        Point pointVec2 = Point(p.x - mean.x, p.y - mean.y);



        double distance = min(findDistance(mainVec1, pointVec1), findDistance(mainVec2, pointVec2));

        if (distance > max_distance) {
            max_distance = distance;
        }
    }

    return max_distance < eps;
}

vector<vector<Point>> my_Aproxx(vector<vector<Point>> sectors,int minSectorSize= 50, double eps = 10)
{

    vector<vector<Point>> aproxx_sectors_vector = vector<vector<Point>>();

    for (auto& sec : sectors) {

        vector<Point> aproxx_sector = vector<Point>();

        if (sec.size() < minSectorSize) {
            continue;
        }

        Point first = sec[0];
        Point max_point = findMaxDistancePoint(first, sec);

        Point second_max_point = findMaxDistancePoint(max_point, sec);

        max_point = findMaxDistancePoint(second_max_point, sec);

        if (max_point.x < second_max_point.x) {
            aproxx_sector.push_back(max_point);
            aproxx_sector.push_back(second_max_point);
        }
        else {
            aproxx_sector.push_back(second_max_point);
            aproxx_sector.push_back(max_point);   
        }
        
        //aproxx_sectors_vector.push_back(aproxx_sector);



        vector<Point> upperPoints = vector<Point>();
        vector<Point> lowerPoints = vector<Point>();

        Point diogVec = Point(aproxx_sector[1].x - aproxx_sector[0].x,
                              aproxx_sector[1].y - aproxx_sector[0].y);
        
        for (const auto& p : sec) {


            Point pointVec = Point(p.x - aproxx_sector[0].x,
                                   p.y - aproxx_sector[0].y);

            int vectorProduct = diogVec.x * pointVec.y - diogVec.y * pointVec.x;

            if (vectorProduct > 0) {
                upperPoints.push_back(p);

            }
            else {
                lowerPoints.push_back(p);
            }

        }

        auto [max_point_u, max_d1] = findMaxPerpendicularPoint(aproxx_sector[0], aproxx_sector[1], upperPoints);
        auto [max_point_l, max_d2] = findMaxPerpendicularPoint(aproxx_sector[0], aproxx_sector[1], lowerPoints);

        bool flag = true;

        if (isSquare(aproxx_sector[0], max_point_u, aproxx_sector[1], upperPoints, eps) &&
            isSquare(aproxx_sector[0], max_point_l, aproxx_sector[1], lowerPoints, eps) &&
            max_d1 > minSectorSize / 10 && max_d2 > minSectorSize / 10)
        {
            aproxx_sector.push_back(max_point_u);
            aproxx_sector.push_back(max_point_l);
            cout << aproxx_sector;
            aproxx_sectors_vector.push_back(aproxx_sector);



        }
    }

    


    return aproxx_sectors_vector;


}

Mat Marker(const Mat& input,vector<Point> aproxx_sector, int num_of_parts = 240) {

    Mat result = Mat::zeros(num_of_parts, num_of_parts, CV_8UC1);

    Point p1 = aproxx_sector[3];
    Point p2 = aproxx_sector[1];
    Point p3 = aproxx_sector[0];
    Point p4 = aproxx_sector[2];

    double up_dx = (p2.x - p1.x) ;
    double up_dy = (p2.y - p1.y);

    double down_dx = (p4.x - p3.x);
    double down_dy = (p4.y - p3.y);

    for (int i = 0; i < num_of_parts; i++) {

        Point2d up_point = Point2d(p1.x + i * up_dx / num_of_parts, p1.y + i * up_dy / num_of_parts);
        Point2d down_point = Point2d(p3.x + i * down_dx / num_of_parts, p3.y + i * down_dy / num_of_parts);


        double dx = (up_point.x - down_point.x);
        double dy = (up_point.y - down_point.y);


        for (int j = 0; j < num_of_parts; j++) {

            int x = round(down_point.x + j * dx / num_of_parts);
            int y = round(down_point.y + j * dy / num_of_parts);

            x = max(min(x, input.cols - 1), 0);
            y = max(min(y, input.rows - 1), 0);


            result.at<uchar>(i, j) = input.at<uchar>(y, x);
        }
    }
    return result;
}

bool isMarker(const cv::Mat& marker, int num_of_parts = 240, int gridSize = 6) {

    if (marker.rows < 10 || marker.cols < 10) {
        return false;
    }
    // Размер маркера 6x6 с границей 1 пиксель (итого 8x8)
    cv::Mat resizedMarker = cv::Mat::zeros(gridSize, gridSize, CV_8UC1);

    // Размер каждого сектора
    float cellWidth = num_of_parts / gridSize;
    float cellHeight = num_of_parts / gridSize;

    // Определение цвета каждого сектора
    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            int startX = static_cast<int>(j * cellWidth);
            int startY = static_cast<int>(i * cellHeight);
            int endX = static_cast<int>(startX + cellWidth);
            int endY = static_cast<int>(startY + cellHeight);

            cv::Mat cell = marker(cv::Rect(startX, startY, endX - startX, endY - startY));

            // Подсчет черных и белых пикселей в ячейке
            int blackCount = cv::countNonZero(cell == 0);
            int whiteCount = cv::countNonZero(cell == 255);

            // Определение доминирующего цвета
            if (blackCount > whiteCount) {
                resizedMarker.at<uchar>(i, j) = 0; // Черный цвет
            }
            else {
                resizedMarker.at<uchar>(i, j) = 255; // Белый цвет
            }
        }
    }

    // Рисование линий для разграничения зон проверки на исходном маркере
    cv::Mat markerWithGrid = marker.clone();
    for (int i = 1; i < gridSize; ++i) {
        cv::line(marker, cv::Point(static_cast<int>(i * cellWidth), 0), cv::Point(static_cast<int>(i * cellWidth), marker.rows), cv::Scalar(128), 1);
        cv::line(marker, cv::Point(0, static_cast<int>(i * cellHeight)), cv::Point(marker.cols, static_cast<int>(i * cellHeight)), cv::Scalar(128), 1);
    }


    // Проверка рамки маркера (рамка должна быть черной)
    for (int i = 0; i < gridSize; ++i) {
        if (resizedMarker.at<uchar>(0, i) != 0 || resizedMarker.at<uchar>(gridSize - 1, i) != 0 ||
            resizedMarker.at<uchar>(i, 0) != 0 || resizedMarker.at<uchar>(i, gridSize - 1) != 0) {
            return false;
        }
    }

    return true;
}

int fun(vector<Point> marker_points, Point3d R, Parameters camera_parameters) {



    double fx = camera_parameters.cameraMatrix.at<double>(0, 0);
    double fy = camera_parameters.cameraMatrix.at<double>(1, 1);
    double cx = camera_parameters.cameraMatrix.at<double>(0, 2);
    double cy = camera_parameters.cameraMatrix.at<double>(1, 2);


    Point2d p1 = Point2d(marker_points[3]);
    Point2d p2 = Point2d(marker_points[1]);
    Point2d p3 = Point2d(marker_points[0]);
    Point2d p4 = Point2d(marker_points[2]);
    

    vector<Point2d> imagePoints = { p1, p2, p3, p4 };
    double x_mean = 0;
    double y_mean = 0;
    for(auto p : imagePoints) {
        x_mean += p.x;
        y_mean += p.y;
    }
    imagePoints.push_back(Point2d(x_mean / 4, y_mean / 4));

    double len = camera_parameters.arucoSquareDimension;

    vector<Point2d> realPoints = vector<Point2d>();
    realPoints.push_back(Point2d(- len / 2, len / 2));
    realPoints.push_back(Point2d(len / 2, len / 2));
    realPoints.push_back(Point2d(- len / 2, - len / 2));
    realPoints.push_back(Point2d(len / 2, len / 2));
    realPoints.push_back(Point2d(0, 0));

    vector<double> x = {};
    vector<double> y = {};
    vector<double> u = {};
    vector<double> v = {};

    for (int i = 0; i < imagePoints.size(); i++) {
        x.push_back(realPoints[i].x);
        y.push_back(realPoints[i].y);
        u.push_back(imagePoints[i].x);
        v.push_back(imagePoints[i].y);
    }

    int N = x.size();
    MatrixXd A(2 * N, 9);
    VectorXd b(2 * N);

    for (int i = 0; i < imagePoints.size(); i++) {
        x.push_back(realPoints[i].x);
        y.push_back(realPoints[i].y);
        u.push_back(imagePoints[i].x);
        v.push_back(imagePoints[i].y);
    }

    for (int i = 0; i < N; ++i) {
        double xi = x[i], yi = y[i], ui = u[i], vi = v[i];

        A(2 * i, 0) = fx * xi;
        A(2 * i, 1) = fx * yi;
        A(2 * i, 2) = fx;
        A(2 * i, 3) = 0;
        A(2 * i, 4) = 0;
        A(2 * i, 5) = 0;
        A(2 * i, 6) = cx * xi;
        A(2 * i, 7) = cx * yi;
        A(2 * i, 8) = cx;

        A(2 * i + 1, 0) = 0;
        A(2 * i + 1, 1) = 0;
        A(2 * i + 1, 2) = 0;
        A(2 * i + 1, 3) = fy * xi;
        A(2 * i + 1, 4) = fy * yi;
        A(2 * i + 1, 5) = fy;
        A(2 * i + 1, 6) = cy * xi;
        A(2 * i + 1, 7) = cy * yi;
        A(2 * i + 1, 8) = cy;

        b(2 * i) = ui;
        b(2 * i + 1) = vi;
    }

    std::cout << A << endl;
    VectorXd p = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

    std::cout << p.transpose() << endl;
    return 0;
}

int main() {
    




    Parameters param;
    param.arucoSquareDimension = 0.25f; //длина стороны маркера ArUco м
    param.cameraMatrix = (Mat_<double>(3, 3) << 10000, 0, 0, 0, 10000, 0, 0, 0, 1); // Внутренние параметры камеры (фокус и смещение)
    param.distanceCoefficients = (Mat_<double>(1, 5) << -0, 0, -0, -0, -0); // Коэффициенты дисторсии

    // Матрица камеры (cameraMatrix)          Коэффициенты дисторсии (distCoefficients):
    // [ fx   0  cx ]                           [ k1, k2, p1, p2, k3 ]q
    // [  0  fy  cy ]                           k1, k2, k3 - коэффициенты радиального искажения
    // [  0   0   1 ]                           p1, p2 - коэффициенты тангенциального искажения
    // Флаг для выбора: true для калибровки, false для загрузки из файла
    //bool recalibrate = false;
    //string calibrationFile = "cameraCalibrationData";
    ////Вызов функции для калибровки камеры или загрузки параметров из файла ()
    //calibrateOrLoadCamera(recalibrate, calibrationFile, param.boardSize, param.squareEdgeLength, param.cameraMatrix, param.distanceCoefficients);





    vector<Point> input_P = { Point(907, 516), Point(978, 642), Point(880, 615), Point(1005, 541) };
    
    waitKey();

    string inputImagePath = "..\\..\\..\\data\\IMM1.jpeg";
    Mat inputImage = imread(inputImagePath);
    cv::imshow("Input image", inputImage);


    Mat image = inputImage.clone();
    cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
    Mat bin= getBinaryImage(image, 128);
    cv::imshow("Input image", bin);



    imwrite("..\\..\\..\\data\\outbin.jpg", bin);
    cv::waitKey();


    Mat Cx = Border(bin);
    imwrite("..\\..\\..\\data\\out.jpg", Cx);
    cv::imshow("Input image", Cx);
    cv::waitKey();

    vector<vector<Point>> sectors_vector = Sectors(Cx);
    vector<vector<Point>> aproxx_sectors_vector = my_Aproxx(sectors_vector, 50, 5);
    vector<vector<Point>> result_sectors_vector = vector<vector<Point>>();
    for (auto sec : aproxx_sectors_vector) {

        Mat mark = Marker(bin, sec);
        imshow("Marker", mark);
        waitKey();
        if (isMarker(mark, 240, 8)) {
            result_sectors_vector.push_back(sec);
            fun(sec, Point3d(0, 0, 0), param);
            imshow("Marker2", mark);
            waitKey();
        }
        
        circle(inputImage, sec[0], 5, Scalar(255, 0, 0));
        
    }

    cv::imshow("Input image", image);
    cv::waitKey();


    return 0;

}