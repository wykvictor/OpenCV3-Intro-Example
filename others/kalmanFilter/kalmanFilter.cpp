//-------------------------------------------------------------------------------------------------
//      描述：Kalman Filter
//-------------------------------------------------------------------------------------------------
#include <opencv2/opencv.hpp>
using namespace cv;

const int winHeight = 800;
const int winWidth = 1000;
cv::Point mouse_info = Point(0,0);

void on_MouseHandle(int event, int x, int y, int flags, void *param);
void DrawRectangle(cv::Mat &img, cv::Rect box);

KalmanFilter KF(4, 2, 0);

int main(int argc, char **argv) {
  KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
  // 1. init...
  KF.statePre.at<float>(0) = mouse_info.x;
  KF.statePre.at<float>(1) = mouse_info.y;
  KF.statePre.at<float>(2) = 0;
  KF.statePre.at<float>(3) = 0;
  setIdentity(KF.measurementMatrix);
  // http://campar.in.tum.de/Chair/KalmanFilter
  setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
  setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
  setIdentity(KF.errorCovPost, Scalar::all(.1));

  Mat srcImage(600, 800, CV_8UC3);
  srcImage = Scalar::all(255);

  namedWindow("Kalman");
  setMouseCallback("Kalman", on_MouseHandle, (void *)&srcImage);
  Point old_predict_pt = mouse_info, old_mouse = mouse_info;
  while (1) {
    //2.kalman prediction  
    Mat prediction = KF.predict();
    Point predict_pt = Point(prediction.at<float>(0), prediction.at<float>(1));   //预测值(x',y')  
    Mat_<float> measurement(2, 1); measurement.setTo(Scalar(0));                  //3.update measurement  
    measurement.at<float>(0) = (float)mouse_info.x;
    measurement.at<float>(1) = (float)mouse_info.y;

    //4.update  
    KF.correct(measurement);

    //draw   
    //srcImage = Scalar::all(255);
    //circle(srcImage, predict_pt, 5, Scalar(0, 255, 0), 3);    //predicted point with Green  
    //circle(srcImage, mouse_info, 5, Scalar(255, 0, 0), 3);    //current position with Blue     
    line(srcImage, old_predict_pt, predict_pt, Scalar(0, 255, 0), 1, 16);
    line(srcImage, old_mouse, Point(measurement.at<float>(0), measurement.at<float>(1)), Scalar(255, 0, 0), 1, 16);

    char buf[256];
    sprintf_s(buf, 256, "predicted position:(%3d,%3d)", predict_pt.x, predict_pt.y);
    //putText(srcImage, buf, Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 1, 8);
    sprintf_s(buf, 256, "current position :(%3d,%3d)", mouse_info.x, mouse_info.y);
    //putText(srcImage, buf, cvPoint(10, 60), CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 1, 8);

    imshow("Kalman", srcImage);
    if (waitKey(20) == 27) break;  //按下ESC键，程序退出
    old_predict_pt = predict_pt;
    old_mouse = Point(measurement.at<float>(0), measurement.at<float>(1));
  }
  return 0;
}

//--------------------------------【on_MouseHandle( )函数】-----------------------------
//      描述：鼠标回调函数，根据不同的鼠标事件进行不同的操作
//-----------------------------------------------------------------------------------------------
void on_MouseHandle(int event, int x, int y, int flags, void* param) {
    Mat& image = *(cv::Mat*) param;
    if (event == CV_EVENT_MOUSEMOVE) {
      mouse_info = Point(x, y);
    }
}