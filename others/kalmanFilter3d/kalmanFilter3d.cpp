//-------------------------------------------------------------------------------------------------
//      描述：Kalman Filter
//-------------------------------------------------------------------------------------------------
#include <opencv2/opencv.hpp>
using namespace cv;

struct Tag {
  int id = 0;                          // id of the tag
  float x = 0, y = 0, z = 0;           // translation matrix of tag relative to camera
  float yaw = 0, pitch = 0, roll = 0;  // rotation matrix of tag relative to camera
  int confidence = 0;  // read-only for tag and object, larger value means more confident result
};

const int winHeight = 800;
const int winWidth = 1000;
cv::Point mouse_info = Point(0,0);

void on_MouseHandle(int event, int x, int y, int flags, void *param);
void DrawRectangle(cv::Mat &img, cv::Rect box);

class Kalman {
public:
  Kalman() { init(); }
  void estimate(Tag &tagToEstimate) {
    // fill measurements
    cv::Mat_<double> measurement(6, 1);
    measurement.at<double>(0) = tagToEstimate.x;
    measurement.at<double>(1) = tagToEstimate.y;
    measurement.at<double>(2) = tagToEstimate.z;
    measurement.at<double>(3) = tagToEstimate.roll;
    measurement.at<double>(4) = tagToEstimate.pitch;
    measurement.at<double>(5) = tagToEstimate.yaw;
    // First predict, to update the internal statePre variable
    cv::Mat prediction = KF.predict();

    // The "correct" phase that is going to use the predicted value and our measurement
    cv::Mat estimated = KF.correct(measurement);

    // Estimated translation
    tagToEstimate.x = estimated.at<double>(0);
    tagToEstimate.y = estimated.at<double>(1);
    tagToEstimate.z = estimated.at<double>(2);

    // Estimated euler angles
    tagToEstimate.roll = estimated.at<double>(9);
    tagToEstimate.pitch = estimated.at<double>(10);
    tagToEstimate.yaw = estimated.at<double>(11);
  }
  void setTransitionMatrix(cv::Mat &matrix, double dt) {

    /** DYNAMIC MODEL **/

    //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]

    // position
    matrix.at<double>(0, 3) = dt;
    matrix.at<double>(1, 4) = dt;
    matrix.at<double>(2, 5) = dt;
    matrix.at<double>(3, 6) = dt;
    matrix.at<double>(4, 7) = dt;
    matrix.at<double>(5, 8) = dt;
    matrix.at<double>(0, 6) = 0.5 * pow(dt, 2);
    matrix.at<double>(1, 7) = 0.5 * pow(dt, 2);
    matrix.at<double>(2, 8) = 0.5 * pow(dt, 2);

    // orientation
    matrix.at<double>(9, 12) = dt;
    matrix.at<double>(10, 13) = dt;
    matrix.at<double>(11, 14) = dt;
    matrix.at<double>(12, 15) = dt;
    matrix.at<double>(13, 16) = dt;
    matrix.at<double>(14, 17) = dt;
    matrix.at<double>(9, 15) = 0.5 * pow(dt, 2);
    matrix.at<double>(10, 16) = 0.5 * pow(dt, 2);
    matrix.at<double>(11, 17) = 0.5 * pow(dt, 2);
  }

private:
  void init() {
    KF.init(nStates, nMeasurements, nInputs, CV_64F);  // init Kalman Filter

    setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));      // set process noise
    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-4));  // set measurement noise
    setIdentity(KF.errorCovPost, cv::Scalar::all(.1));            // error covariance

    setTransitionMatrix(KF.transitionMatrix, dt);

    /** MEASUREMENT MODEL **/

    //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]

    KF.measurementMatrix.at<double>(0, 0) = 1;   // x
    KF.measurementMatrix.at<double>(1, 1) = 1;   // y
    KF.measurementMatrix.at<double>(2, 2) = 1;   // z
    KF.measurementMatrix.at<double>(3, 9) = 1;   // roll
    KF.measurementMatrix.at<double>(4, 10) = 1;  // pitch
    KF.measurementMatrix.at<double>(5, 11) = 1;  // yaw
  }

  cv::KalmanFilter KF;

  int nStates = 18;       // the number of states
  int nMeasurements = 6;  // the number of measured states
  int nInputs = 0;        // the number of action control

  double dt = 0.02;  // time between measurements (1/FPS)
};

int main(int argc, char **argv) {
  Kalman KF;         // instantiate Kalman Filter

  Mat srcImage(600, 800, CV_8UC3);
  srcImage = Scalar::all(255);

  namedWindow("Kalman");
  setMouseCallback("Kalman", on_MouseHandle, (void *)&srcImage);
  Point old_predict_pt = mouse_info, old_mouse = mouse_info;
  while (1) {
    float mouse_x = (float)mouse_info.x;
    float mouse_y = (float)mouse_info.y;
    //2.kalman prediction
    Tag obj_pose;
    obj_pose.x = mouse_x;
    obj_pose.y = mouse_y;
    obj_pose.z = 1;
    obj_pose.roll = 0;
    obj_pose.pitch = 0;
    obj_pose.yaw = 0;
    KF.estimate(obj_pose);
    Point predict_pt = Point(obj_pose.x, obj_pose.y);   //预测值(x',y')  

    //draw   
    //srcImage = Scalar::all(255);
    //circle(srcImage, predict_pt, 5, Scalar(0, 255, 0), 3);    //predicted point with Green  
    //circle(srcImage, mouse_info, 5, Scalar(255, 0, 0), 3);    //current position with Blue     
    line(srcImage, old_predict_pt, predict_pt, Scalar(0, 255, 0), 1, 16);
    line(srcImage, old_mouse, Point(mouse_x, mouse_y), Scalar(255, 0, 0), 1, 16);

    char buf[256];
    sprintf_s(buf, 256, "predicted position:(%3d,%3d)", predict_pt.x, predict_pt.y);
    //putText(srcImage, buf, Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 1, 8);
    sprintf_s(buf, 256, "current position :(%3d,%3d)", mouse_info.x, mouse_info.y);
    //putText(srcImage, buf, cvPoint(10, 60), CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 1, 8);

    imshow("Kalman", srcImage);
    if (waitKey(20) == 27) break;  //按下ESC键，程序退出
    old_predict_pt = predict_pt;
    old_mouse = Point(mouse_x, mouse_y);
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