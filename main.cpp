/**
    Opencv example code: camera image rectification from a chessboard (computing homography matrix)
    Enrique Marin
    88enrique@gmail.com
*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(){

    // Variables
    VideoCapture capture;
    Mat rectified, frame;

    namedWindow("video", CV_WINDOW_AUTOSIZE);
    namedWindow("rectified", CV_WINDOW_AUTOSIZE);

    // Open video file
    capture.open("../Videos/chessboard-1.avi");
    //capture.open(0);

    // Check that the video was opened
    if (!capture.isOpened()){
        cout << "Cannot open video device or file!" << endl;
        return -1;
    }

    // Read the video
    while(true){

        // Read new frame
        capture.read(frame);
        if (frame.empty())
            break;

        // Chessboard size
        Size patternsize(9,6); //interior number of corners
        vector<Point2f> corners; //this will be filled by the detected corners

        // Find corners on a chessboard pattern
        bool patternfound = findChessboardCorners(frame, patternsize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
        //drawChessboardCorners(frame, patternsize, Mat(corners), patternfound);

        rectified = frame.clone();
        if (patternfound){

            // Correspondences from object to world
            vector<Point2f> img;
            vector<Point2f> world;
            world.push_back(Point2f(450,300));
            img.push_back(corners.at(0));
            world.push_back(Point2f(50,300));
            img.push_back(corners.at(8));
            world.push_back(Point2f(450,50));
            img.push_back(corners.at(45));
            world.push_back(Point2f(50,50));
            img.push_back(corners.at(53));

            // Camera matrix
            double k[] = {1,0, -100, 0, 1, -100, 0, 0 ,1};
            Mat K = Mat(3, 3, CV_64FC1, k);

            // Find homography matrix
            // H:world->image; H-1:image->world
            Mat H = findHomography(world, img, CV_RANSAC);
            H = H*K;
            rectified = Mat(frame.cols,frame.rows, CV_32FC3);

            // Apply rectification to the image
            warpPerspective(frame, rectified, H.inv(), cvSize(rectified.rows, rectified.cols));

            // Paint 4 corners used in rectification (frame image)
            circle(frame, corners.at(0), 10, cvScalar(0,255,0));
            circle(frame, corners.at(8), 10, cvScalar(0,255,0));
            circle(frame, corners.at(45), 10, cvScalar(0,255,0));
            circle(frame, corners.at(53), 10, cvScalar(0,255,0));

            // Transform points from image to rectified image
            vector<Point2f> img_points;     // Points in image
            vector<Point2f> world_points;   // Points rectified
            img_points.push_back(corners.at(0));
            img_points.push_back(corners.at(8));
            img_points.push_back(corners.at(45));
            img_points.push_back(corners.at(53));

            // Apply homography (rectification)
            perspectiveTransform(img_points, world_points, H.inv());

            // Paint 4 corners on rectified image
            circle(rectified, world_points.at(0), 10, cvScalar(0,255,0));
            circle(rectified, world_points.at(1), 10, cvScalar(0,255,0));
            circle(rectified, world_points.at(2), 10, cvScalar(0,255,0));
            circle(rectified, world_points.at(3), 10, cvScalar(0,255,0));

            // Show rectified image
            imshow("rectified", rectified);
        }

        // Show frame
        imshow("video", frame);

        if ((cvWaitKey(10) & 255) == 27) break;
    }

    // Release Mat memory
    rectified.release();
    frame.release();

    return 0;
}


