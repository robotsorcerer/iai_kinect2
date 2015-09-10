/**
 * Copyright 2014 University of Bremen, Institute for Artificial Intelligence
 * Author: Thiemo Wiedemeyer <wiedemeyer@cs.uni-bremen.de>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <fcntl.h>      //unix headers
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <mutex>
#include <thread>
#include <chrono>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <ros/ros.h>
#include <ros/spinner.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>  // convert between opencv images and ros msgs
#include <sensor_msgs/Image.h>

#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <kinect2_bridge/kinect2_definitions.h>

#include "savgol.h"
#include <iostream>
#include <cmath>


using namespace std;
using namespace cv;
using namespace Eigen;
using namespace cv::gpu;

/** Global variables */
const String face_cascade_name = "haarcascade_frontalface_alt.xml";
const String eyes_cascade_name = "haarcascade_eye.xml";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

CascadeClassifier_GPU face_cascade_gpu;
CascadeClassifier_GPU eyes_cascade_gpu;

RNG rng(12345);

//font properties
static const cv::Point pos(5, 15);
static const cv::Scalar colorText = CV_RGB(0, 0, 255);
static const double sizeText = 0.5;
static const int lineText = 1;
static const int font = cv::FONT_HERSHEY_SIMPLEX;

KalmanFilter KF(2, 1, 0);
KalmanFilter KF2(2,1, 0);
Mat processNoise(2, 1, CV_32F);
Mat state(2, 1, CV_32F);

const int F = 5;      //Frame Size
const int k = 3;      //Example Polynomial Order
const double Fd = (double) F;        //sets the frame size for the savgol differentiation coefficients. This must be odd

uint16_t rosdepth;            
  
const float Qt = 1500.0;          //cov matrices for kalman filter
const float Rt = 30;

const float Rt2 = 4.6325;         //cov matrix for kalman filter 2

std::ostringstream oss;

/** Function Prototypes */
void detectAndDisplay( Mat detframe, Mat depth );
//void talker(Mat& rosobs, Mat& rospred, Mat& rosupd, Mat& rospred_error, Mat& rosest_error, Mat& rosgain);
void talker(float& rosobs, float& rospred, float& rosupd, float& rospred_error, float& rosest_error, float& rosgain);
void kalman(float deltaT, Mat measurement);
//void kalman2(float& deltaT,float& rosupd);
MatrixXi vander(const int F);     
MatrixXf sgdiff(int k, double Fd);
RowVectorXf savgolfilt(VectorXf x, VectorXf x_on, int k, int F, MatrixXf DIM);
void savgol(float rosupd);

class Receiver
{
public:
  enum Mode
  {
    IMAGE = 0,
    CLOUD,
    BOTH
  };

private:
  std::mutex lock;

  const std::string topicColor, topicDepth;
  const bool useExact, useCompressed;

  bool updateImage, updateCloud;
  bool save;
  bool running;
  size_t frame;
  const size_t queueSize;

  cv::Mat color, depth;
  cv::Mat cameraMatrixColor, cameraMatrixDepth;
  cv::Mat lookupX, lookupY;

  int eye_x, eye_y;
  uchar table;

  typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactSyncPolicy;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ApproximateSyncPolicy;

  ros::NodeHandle nh;             //starts the ROS cpp node, the 1st calls ros::start() and the last calls ros::shutdown()
  ros::AsyncSpinner spinner;
  image_transport::ImageTransport it;
  image_transport::SubscriberFilter *subImageColor, *subImageDepth;
  message_filters::Subscriber<sensor_msgs::CameraInfo> *subCameraInfoColor, *subCameraInfoDepth;

  message_filters::Synchronizer<ExactSyncPolicy> *syncExact;
  message_filters::Synchronizer<ApproximateSyncPolicy> *syncApproximate;

  std::thread imageViewerThread;
  Mode mode;

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
  pcl::PCDWriter writer;
  std::vector<int> params;

public:
  Receiver(const std::string &topicColor, const std::string &topicDepth, const bool useExact, const bool useCompressed)
    : topicColor(topicColor), topicDepth(topicDepth), useExact(useExact), useCompressed(useCompressed),
      updateImage(false), updateCloud(false), save(false), running(false), frame(0), queueSize(5),
      nh("~"), spinner(0), it(nh), mode(CLOUD)
  {
    cameraMatrixColor = cv::Mat::zeros(3, 3, CV_64F);
    cameraMatrixDepth = cv::Mat::zeros(3, 3, CV_64F);
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(100);
    params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    params.push_back(1);
    params.push_back(cv::IMWRITE_PNG_STRATEGY);
    params.push_back(cv::IMWRITE_PNG_STRATEGY_RLE);
    params.push_back(0);
  }

  ~Receiver()
  {
  }

  void run(const Mode mode)
  {
    start(mode);
    stop();
  }

private:
  void start(const Mode mode)
  {
    this->mode = mode;
    running = true;

    std::string topicCameraInfoColor = topicColor.substr(0, topicColor.rfind('/')) + "/camera_info";
    std::string topicCameraInfoDepth = topicDepth.substr(0, topicDepth.rfind('/')) + "/camera_info";

    image_transport::TransportHints hints(useCompressed ? "compressed" : "raw");
    subImageColor = new image_transport::SubscriberFilter(it, topicColor, queueSize, hints);
    subImageDepth = new image_transport::SubscriberFilter(it, topicDepth, queueSize, hints);
    subCameraInfoColor = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoColor, queueSize);
    subCameraInfoDepth = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoDepth, queueSize);

    if(useExact)
    {
      syncExact = new message_filters::Synchronizer<ExactSyncPolicy>(ExactSyncPolicy(queueSize), *subImageColor, *subImageDepth, *subCameraInfoColor, *subCameraInfoDepth);
      syncExact->registerCallback(boost::bind(&Receiver::callback, this, _1, _2, _3, _4));
    }
    else
    {
      syncApproximate = new message_filters::Synchronizer<ApproximateSyncPolicy>(ApproximateSyncPolicy(queueSize), *subImageColor, *subImageDepth, *subCameraInfoColor, *subCameraInfoDepth);
      syncApproximate->registerCallback(boost::bind(&Receiver::callback, this, _1, _2, _3, _4));
    }

    spinner.start();

    std::chrono::milliseconds duration(1);
    while(!updateImage || !updateCloud)
    {
      if(!ros::ok())
      {
        return;
      }
      std::this_thread::sleep_for(duration);
    }
    cloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
    cloud->height = color.rows;
    cloud->width = color.cols;
    cloud->is_dense = false;
    cloud->points.resize(cloud->height * cloud->width);
    createLookup(this->color.cols, this->color.rows);

    switch(mode)
    {
    case CLOUD:
      cloudViewer();
      break;
    case IMAGE:
      imageViewer();
      break;
    case BOTH:
      imageViewerThread = std::thread(&Receiver::imageViewer, this);
      cloudViewer();
      break;
    }
  }

  void stop()
  {
    spinner.stop();

    if(useExact)
    {
      delete syncExact;
    }
    else
    {
      delete syncApproximate;
    }

    delete subImageColor;
    delete subImageDepth;
    delete subCameraInfoColor;
    delete subCameraInfoDepth;

    running = false;
    if(mode == BOTH)
    {
      imageViewerThread.join();
    }
  }

  void callback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth,
                const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor, const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth)
  {
    cv::Mat color, depth;

    readCameraInfo(cameraInfoColor, cameraMatrixColor);
    readCameraInfo(cameraInfoDepth, cameraMatrixDepth);
    readImage(imageColor, color);
    readImage(imageDepth, depth);

    // IR image input
    if(color.type() == CV_16U)
    {
      cv::Mat tmp;
      color.convertTo(tmp, CV_8U, 0.02);
      cv::cvtColor(tmp, color, CV_GRAY2BGR);
    }

    lock.lock();
    this->color = color;
    this->depth = depth;
    updateImage = true;
    updateCloud = true;
    lock.unlock();
  }

  /*Compute the polynomial basis vectors s_0, s_1, s_2 ... s_n using the vandermonde matrix.*/
  MatrixXi vander(const int F)
  {
    VectorXi v = VectorXi::LinSpaced(F,(-(F-1)/2),((F-1)/2)).transpose().eval();

    MatrixXi A(F, F+1);     //We basically compute an F X F+1 matrix;

    for(int i = 0; i < F; ++ i)
    {
     for(int j=1; j < F+1; ++j)
      {
       A(i,j) = pow(v(i), (j-1) ); 
      }
    }

    A = A.block(0, 1, F, F );   //and retrieve the right F X F matrix block, excluding the first column block to find the vandermonde matrix.

    return A;
  }
  
  /*Compute the S-Golay Matrix of differentiators*/
MatrixXf sgdiff(int k, double Fd)
{
  //We set the weighting matrix to an identity matrix if no weighting matrix is supplied
  MatrixXf W = MatrixXf::Identity(Fd, Fd);      

  //Compute Projection Matrix B
  MatrixXi s = vander(F);   

  //Retrieve the rank deficient matrix from the projection matrix
  MatrixXi S = s.block(0, 0, s.rows(), (k+1) ) ; 

  //Compute sqrt(W)*S
  MatrixXf Sd = S.cast<float> ();    //cast S to float
  MatrixXf inter = W * Sd;              //W is assumed to be identity. Change this if you have reasons to.

  //Compute the QR Decomposition
  HouseholderQR<MatrixXf> qr(inter);
  qr.compute(inter);

  FullPivLU<MatrixXf>lu_decomp(inter);      //retrieve rank of matrix
  
  int Rank = lu_decomp.rank() ;
   
  //For rank deficient matrices. The S matrix block will always be rank deficient.        
  MatrixXf Q = qr.householderQ();
  MatrixXf R = qr.matrixQR().topLeftCorner(Rank, Rank).template triangularView<Upper>();

  //Compute Matrix of Differentiators
  MatrixXf Rinv = R.inverse();
  MatrixXf RinvT = Rinv.transpose();

  MatrixXf G = Sd * Rinv * RinvT;           /*G = S(S'S)^(-1)   -- eqn 8.3.90 (matrix of differentiation filters)*/
 
  MatrixXf SdT = Sd.transpose().eval();

  MatrixXf B = G * SdT * W;   //SG-Smoothing filters of length F and polynomial order k

  return B;
}

RowVectorXf savgolfilt(VectorXf x, VectorXf x_on, int k, int F)
{  
  Matrix4f DIM = Matrix4f::Zero();        //initialize DIM as a matrix of zeros if it is not supplied
  int siz = x.size();       //Reshape depth values by working along the first non-singleton dimension

  //Find leading singleton dimensions
  
  MatrixXf B = sgdiff(k, Fd);       //retrieve matrix B

  /*Transient On*/
  int id_size = (F+1)/2 - 1;
  MatrixXf Bbutt = B.bottomLeftCorner((F-1)/2, B.cols());

  int n = Bbutt.rows();
  //flip Bbutt from top all the way down 
  MatrixXf Bbuttflipped(n, Bbutt.cols());
 
    for(int j = n - 1; j >= 0;)
    { 
      for(int i = 0; i < n ; ++i)
      {        
        Bbuttflipped.row(i) = Bbutt.row(j);
        --j;
      }
    }
    
  //flip x_on up and down as above
  VectorXf x_onflipped(x_on.rows(), x_on.cols());  //pre-allocate
  x_onflipped.transpose().eval();     

  int m = x_on.size();                          //retrieve total # coefficients

    for(int j = m -1; j >=0;)
    {
      for(int i = 0; i < m; ++i)
      {
        x_onflipped.row(i) = x_on.row(j);
        --j;
      }
    }
  
  VectorXf y_on = Bbuttflipped * x_onflipped;  //Now compute the transient on

 /*Compute the steady state output*/
  size_t idzeroth = floor(B.cols()/2);
  VectorXf Bzeroth = B.col(idzeroth);
  VectorXf Bzerothf = Bzeroth.cast<float>();

  VectorXf y_ss = Bzerothf.transpose().eval() * x;     //This is the steady-state smoothed value

  /*Compute the transient off for non-sequential data*/
  MatrixXf Boff = B.topLeftCorner((F-1)/2, B.cols());

  int p = Boff.rows();                        //flip Boff along the horizontal axis

  MatrixXf Boff_flipped(p, Boff.cols());
    
  for(int j = p - 1; j >= 0;)
  { 
    for(int i = 0; i < p ; ++i)
    {        
      Boff_flipped.row(i) = Boff.row(j);
      --j;
    }
  }

/*x_off will be the last (F-1) x values. Note, if you are smoothing in real time, you need to find 
  a way to let your compiler pick the last F-length samples from your data in order to compute your x_off. 
  You could have the program wait for x_milliseconds before you pick 
  the transient off, for example*/
  VectorXf x_off = VectorXf::LinSpaced(F, x(0), x(F-1)).transpose();  
  VectorXf x_offflipped(x_off.rows(), x_off.cols());      //pre-allocate    
  //flip x_off along the horizontal axis
    int q = x_off.size();                          //retrieve total # coefficients

    for(int j = q -1; j >=0;)
    {
      for(int i = 0; i < q; ++i)
      {
        x_offflipped.row(i) = x_on.row(j);
        --j;
      }
    }
  VectorXf y_off = Boff_flipped * x_offflipped;   //This is the transient off

  /*Make Y into the shape of X and retuen the smoothed values!*/
  RowVectorXf y(F);
  y << y_off.transpose().eval(), y_ss, y_on.transpose().eval();

  return y;
}

  double elapsed;  
  long frmCnt = 0;
  double totalT = 0.0;
  GpuMat faces;
  Mat faces_host;
  GpuMat eyes; 
  Mat eyes_host;
  Mat gray_resized, color_resized;

  void imageViewer()
  {
    cv::Mat color, depth, depthDisp;

    std::chrono::time_point<std::chrono::high_resolution_clock> begin, now;    
    size_t frameCount = 0;
    double fps = 0;
    std::ostringstream oss;

    cv::namedWindow("ROS Features Viewer");
    //cv::namedWindow("Color_resized");
    oss << "starting...";

    Size dsize;
    double fx = 0.5;
    double fy = 0.5;
    int interpol=INTER_LINEAR;

    const double scaleFactor = 1.2;
    const int minNeighbors = 6;

    const Size face_maxSize = Size(20, 20);
    const Size face_minSize = Size(5, 5);

    begin = std::chrono::high_resolution_clock::now();

    for(; running && ros::ok();)
    {
      if(updateImage)
      {
        lock.lock();
        color = this->color;
        depth = this->depth;
        updateImage = false;
        lock.unlock();


        now = std::chrono::high_resolution_clock::now();        
        ++frameCount;
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - begin).count() / 1000.0;
       
        if(elapsed >= 1.0)
        {
          fps = frameCount / elapsed;
          oss.str("");
          oss << "fps: " << fps << " ( " << elapsed / frameCount * 1000.0 << " ms)";
          begin = now;
          frameCount = 0;
        }
        cv::putText(color, oss.str(), Point(20, 55), font, sizeText, colorText, lineText, CV_AA);        
        double t = (double)getTickCount();

        Mat frame_gray;
        cvtColor( color, frame_gray, CV_BGR2GRAY );
        equalizeHist( frame_gray, frame_gray );
        
        resize(frame_gray, gray_resized, Size(), fx, fy, interpol);
        resize(color, color_resized, Size(), fx, fy, interpol);         //for opencv window

        //GpuMat frame_gray_gpu(frame_gray);                  //move grayed color image to gpu
        GpuMat frame_gray_gpu(gray_resized);                  //move grayed color image to gpu
        //cv::putText(color_resized, oss.str(), Point(20, 55), font, sizeText, colorText, lineText, CV_AA);

        //-- Detect faces    
        faces.create(1, 10000, cv::DataType<cv::Rect>::type);   //preallocate gpu faces

        int faces_detect = face_cascade_gpu.detectMultiScale(frame_gray_gpu, faces, face_maxSize, face_minSize, scaleFactor, minNeighbors);

        //Download only detected faces to cpu
        faces.colRange(0, faces_detect).download(faces_host);   

        frame_gray_gpu.release();
        faces.release();

        Rect* cfaces = faces_host.ptr<Rect>();                              // faces result are now in "faces_host"

        t = ( (double)getTickCount() - t)/getTickFrequency();               //measure total time to detect and download

        totalT += t;
        ++frmCnt;

        cout << "fps: " << 1.0 / (totalT / (double)frmCnt) << endl;

        for( int i = 0; i < faces_detect; ++i )
        {    
          std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
          start = chrono::high_resolution_clock::now();

          Point vertex_one ( cfaces[i].x, cfaces[i].y);      
          Point vertex_two ( cfaces[i].x + cfaces[i].width, cfaces[i].y + cfaces[i].height);
          rectangle(color, vertex_one, vertex_two, Scalar(0, 255, 0), 2, 4, 0 );          
          rectangle(color_resized, vertex_one, vertex_two, Scalar(255, 255, 0), 2, 4, 0 );

          Mat faceROI = gray_resized( cfaces[i] );

          GpuMat faceROIGpu(faceROI);                                   //convert faceROI to GpuMat container
         
          int eyes_detect = eyes_cascade_gpu.detectMultiScale(faceROIGpu, eyes, face_maxSize, face_minSize, scaleFactor, minNeighbors);

          cout << "eyes_detect: " << eyes_detect << endl;
          //Download only detected eyes to cpu
          eyes.colRange(0, eyes_detect).download(eyes_host);

          faceROIGpu.release();                           //free memory immediately          
          eyes.release();                                 //free mem immediately

          Rect* ceyes = eyes_host.ptr<Rect>();                        //eyes result now in eyes_host

          for( int j = 0; j < eyes_detect; ++j )
          { 
            Point eye_center( (cfaces[i].x + ceyes[j].x + ceyes[j].width/4), (cfaces[i].y + ceyes[j].y + ceyes[j].height/4) );
            circle( color, eye_center, 4.0, Scalar(255,255,255), CV_FILLED, 8, 0); 
            circle( color_resized, eye_center, 4.0, Scalar(255,255,255), CV_FILLED, 8, 0); 
            rosdepth  = (depth.at<uint16_t>(eye_center.y, eye_center.x) - 932);   //sync the two cames to xbox_values
          }

          end = chrono::high_resolution_clock::now();   
          float deltaT = chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
          Mat measurement = Mat(1, 1, CV_32F, rosdepth);
                
          std::ostringstream osf; 

          osf.str(" ");
          osf <<"Face Point: " << rosdepth << " mm";

          putText(color, osf.str(), Point(20,35), font, sizeText, colorText, lineText,CV_AA);
          putText(color_resized, osf.str(), Point(20,85), font, sizeText, colorText, lineText,CV_AA);
        
          cout <<  "\n\n deltaT: " << deltaT << endl;
          kalman(deltaT, measurement);       //filter observation from kinect          
        }   
      }
        imshow( "ROS Features Viewer", color_resized ); 

      int key = cv::waitKey(1);
      switch(key & 0xFF)
      {
      case 27:
      case 'q':
        running = false;
        break;
      case ' ':
      case 's':
        if(mode == IMAGE)
        {
          createCloud(depth, color, cloud);
          saveCloudAndImages(cloud, color, depth, depthDisp);
        }
        else
        {
          save = true;
        }
        break;
      }
    }
    cv::destroyAllWindows();
    cv::waitKey(100);
  }

  void kalman(float& deltaT, Mat& measurement)
  {
      //Make Q(k) a random walk

      float q11 = pow(deltaT, 4)/4.0 ;
      float q12 = pow(deltaT, 3)/2.0 ;
      float q21 = pow(deltaT, 3)/2.0 ;
      float q22 = pow(deltaT, 2) ;

      KF.processNoiseCov = *(Mat_<float>(2,2) << q11, q12, q21, q22);

      KF.processNoiseCov *=Qt;

      //cout <<"measurement noise cov: " << KF.measurementNoiseCov << endl;

      KF.transitionMatrix = *(Mat_<float>(2, 2) << 1, deltaT, 0, 1);

      Mat prediction = KF.predict(); 

      Mat update =  KF.correct(measurement); 

      float rospred = prediction.at<float>(0);
      float rosupd  = update.at<float>(0);
      float rosobs = measurement.at<float>(0); 

      float rospred_error = KF.errorCovPre.at<float>(0);
      float rosest_error  = KF.errorCovPost.at<float>(0);
      float rosgain       = KF.gain.at<float>(0);

      savgol(rosupd);         //apply savitzky golay to updated measurements
    //  kalman2(deltaT, rosupd);
      talker(rosobs, rospred, rosupd, rospred_error, rosest_error, rosgain) ;      //talk values in a named pipe

      /*  cv::FileStorage fx;
        const String estimates = "ROSPrediction.yaml";
        fx.open(estimates, cv::FileStorage::APPEND);
        fx << "prediction" << rospred;
        fx.release();

        cv::FileStorage fy;
        const String updates = "ROSUpdates.yaml";
        fy.open(updates, cv::FileStorage::APPEND);
        fy << "updates" << rosupd;
        fy.release();

        cv::FileStorage fz;
        const String corrected = "ROSCorrected.yaml";
        fz.open(corrected, cv::FileStorage::APPEND);
        fz << "corrected" << rosobs;
        fz.release(); */
  }
  
  vector<float>updvects;

  void savgol(float& rosupd)
  {
    //collect rosupd into an array of floats with five elements    

   // MatrixXf B = sgdiff(k, Fd);
  /*  temp.pop_back(1);
    temp.push_back(rosupd);
   

    cout <<"\ntemp: " << temp << endl;
 */
    VectorXf x_on = VectorXf::LinSpaced(F, rosupd, rosupd);    //collect the first five values into a matrix

    VectorXf x_shift = push_val(x_on, rosupd);
/*
    cout << "x_on: " << x_on.transpose().eval() <<
            "\nx_shifted: " << x_shift.transpose().eval() << endl;
*/
    //To express as a real filtering operation, we shift x around the nth time instant
    VectorXf x = VectorXf::LinSpaced(F, 700.0, 705.0);

    RowVectorXf Filter = savgolfilt(x, x_on, k, F);

    //cout <<"\n\nFiltered values in the range \n" << x_shift.transpose().eval() <<"\n are: \n" << Filter << endl;
  }

  VectorXf push_val(VectorXf x, float rosupd)
  {
    int num = x.size();
    for(int i = num; i < 0; ++i)
    {
      x(i) = x(i+1);
      x(num - 1) = rosupd;
    }
    return x;
  }
/*
  void kalman2(float& deltaT,float& rosupd)
  {
      Mat measuresq = Mat(1, 1, CV_32F, rosupd);
      //Make Q(k) a random walk
      float q11 = pow(deltaT, 4)/4.0 ;
      float q12 = pow(deltaT, 3)/2.0 ;
      float q21 = pow(deltaT, 3)/2.0 ;
      float q22 = pow(deltaT, 2) ;

      KF2.processNoiseCov = *(Mat_<float>(2,2) << q11, q12, q21, q22);

      KF2.processNoiseCov *=Qt;

      KF2.transitionMatrix = *(Mat_<float>(2, 2) << 1, deltaT, 0, 1);

      Mat prediction2 = KF2.predict(); 

      Mat update2 =  KF2.correct(measuresq); 

      float rospred2 = prediction2.at<float>(0);
      float rosupd2  = update2.at<float>(0);
      float rosobs2 = measuresq.at<float>(0); 


      cout << "obs2: " << rosobs2 <<
              " | pred2: " << rospred2 <<
              " | update2: " << rosupd2 << endl;

      cv::FileStorage fa;
      const String estimates2 = "ROSPrediction2.yaml";
      fa.open(estimates2, cv::FileStorage::APPEND);
      fa << "prediction" << rospred2;
      fa.release();

      cv::FileStorage fb;
      const String updates2 = "ROSUpdates2.yaml";
      fb.open(updates2, cv::FileStorage::APPEND);
      fb << "updates2" << rosupd2;
      fb.release();

      cv::FileStorage fc;
      const String corrected2 = "ROSCorrected2.yaml";
      fc.open(corrected2, cv::FileStorage::APPEND);
      fc << "corrected2" << rosobs2;
      fc.release(); 
  }
*/
  /* Communicate Kalman values in a pipe*/
void talker(float& rosobs, float& rospred, float& rosupd, float& rospred_error, float& rosest_error, float& rosgain)
  {
    int rosfm, rosfp, rosfu;
    int rosfpe, rosfee, rosfg;

   //Measurement FIFO
    const char * rosobsfifo = "/tmp/rosobsfifo";

    mkfifo(rosobsfifo, 0666);                       
    rosfm = open(rosobsfifo, O_WRONLY);         
    write(rosfm, &rosobs, sizeof(rosobs) ); 
    close(rosfm);        

    //Kalman Prediction FIFO
    const char * rospredfifo = "/tmp/rospredfifo";

    mkfifo(rospredfifo, 0666);                       
    rosfp = open(rospredfifo, O_WRONLY);          
    write(rosfp, &rospred, sizeof(rospred) ); 
    close(rosfp);    

    //Kalman Update FIFO
    const char * rosupdfifo = "/tmp/rosupdfifo";
    mkfifo(rosupdfifo, 0666);                       
    rosfu = open(rosupdfifo, O_WRONLY);           
    write(rosfu, &rosupd, sizeof(rosupd) );   
    close(rosfu);

    //Kalman Prediction error FIFO
    const char * rosprederrorfifo = "/tmp/rosprederrorfifo";

    mkfifo(rosprederrorfifo, 0666);                       
    rosfpe = open(rosprederrorfifo, O_WRONLY);           
    write(rosfpe, &rospred_error, sizeof(rospred_error) );   
    close(rosfpe);

    //Kalman Estimation error FIFO
    const char * rosesterrorfifo = "/tmp/rosesterrorfifo";

    mkfifo(rosesterrorfifo, 0666);                       
    rosfee = open(rosesterrorfifo, O_WRONLY);           
    write(rosfee, &rosest_error, sizeof(rosest_error) );   
    close(rosfee);

    //Kalman gain  FIFO
    const char * rosgainfifo = "/tmp/rosgainfifo";

    mkfifo(rosgainfifo, 0666);                       
    rosfg = open(rosgainfifo, O_WRONLY);           
    write(rosfg, &rosgain, sizeof(rosgain) );   
    close(rosfg);

    cout <<"\nrosobs: " << rosobs <<
          "   | rospred: " << rospred <<
          "   | rosupdate: " << rosupd << endl; 
          
    cout << "est_error: " << rosest_error<< 
            " | rpred_error: " << rospred_error << 
            " | gain: "<< rosgain<< endl;
  }

  void cloudViewer()
  {
    cv::Mat color, depth;
    pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
    const std::string cloudName = "rendered";

    lock.lock();
    color = this->color;
    depth = this->depth;
    updateCloud = false;
    lock.unlock();

    createCloud(depth, color, cloud);

    visualizer->addPointCloud(cloud, cloudName);
    visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloudName);
    visualizer->initCameraParameters();
    visualizer->setBackgroundColor(0, 0, 0);
    visualizer->setPosition(mode == BOTH ? color.cols : 0, 0);
    visualizer->setSize(color.cols, color.rows);
    visualizer->setShowFPS(true);
    visualizer->setCameraPosition(0, 0, 0, 0, -1, 0);
    visualizer->registerKeyboardCallback(&Receiver::keyboardEvent, *this);

    for(; running && ros::ok();)
    {
      if(updateCloud)
      {
        lock.lock();
        color = this->color;
        depth = this->depth;
        updateCloud = false;
        lock.unlock();

        createCloud(depth, color, cloud);        
        cv::Mat detframe, depthDisp;
        dispDepth(depth, depthDisp, 12000.0f);        

        visualizer->updatePointCloud(cloud, cloudName);
      }
      if(save)
      {
        save = false;
        cv::Mat depthDisp;
        dispDepth(depth, depthDisp, 12000.0f);
        saveCloudAndImages(cloud, color, depth, depthDisp);
      }
      visualizer->spinOnce(10);
    }
    visualizer->close();
  }

  void keyboardEvent(const pcl::visualization::KeyboardEvent &event, void *)
  {
    if(event.keyUp())
    {
      switch(event.getKeyCode())
      {
      case 27:
      case 'q':
        running = false;
        break;
      case ' ':
      case 's':
        save = true;
        break;
      }
    }
  }

  void readImage(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image) const
  {
    cv_bridge::CvImageConstPtr pCvImage;
    pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
    pCvImage->image.copyTo(image);
  }

  void readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr cameraInfo, cv::Mat &cameraMatrix) const
  {
    double *itC = cameraMatrix.ptr<double>(0, 0);
    for(size_t i = 0; i < 9; ++i, ++itC)
    {
      *itC = cameraInfo->K[i];
    }
  }

  void dispDepth(const cv::Mat &in, cv::Mat &out, const float maxValue)
  {
    cv::Mat tmp = cv::Mat(in.rows, in.cols, CV_8U);
    const uint32_t maxInt = 255;

    #pragma omp parallel for
    for(int r = 0; r < in.rows; ++r)
    {
      const uint16_t *itI = in.ptr<uint16_t>(r);
      uint8_t *itO = tmp.ptr<uint8_t>(r);

      for(int c = 0; c < in.cols; ++c, ++itI, ++itO)
      {
        *itO = (uint8_t)std::min((*itI * maxInt / maxValue), 255.0f);
      }
    }

    cv::applyColorMap(tmp, out, cv::COLORMAP_JET);
  }

  void combine(const cv::Mat &inC, const cv::Mat &inD, cv::Mat &out)
  {
    out = cv::Mat(inC.rows, inC.cols, CV_8UC3);

    #pragma omp parallel for 
    for(int r = 0; r < inC.rows; ++r)
    {
      const cv::Vec3b
      *itC = inC.ptr<cv::Vec3b>(r),
      *itD = inD.ptr<cv::Vec3b>(r);
      cv::Vec3b *itO = out.ptr<cv::Vec3b>(r);

      for(int c = 0; c < inC.cols; ++c, ++itC, ++itD, ++itO)
      {
        itO->val[0] = (itC->val[0] + itD->val[0]) >> 1;
        itO->val[1] = (itC->val[1] + itD->val[1]) >> 1;
        itO->val[2] = (itC->val[2] + itD->val[2]) >> 1;
      }
    }
  }

  void createCloud(const cv::Mat &depth, const cv::Mat &color, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud) const
  {
    const float badPoint = std::numeric_limits<float>::quiet_NaN();

    #pragma omp parallel for
    for(int r = 0; r < depth.rows; ++r)
    {
      pcl::PointXYZRGBA *itP = &cloud->points[r * depth.cols];
      const uint16_t *itD = depth.ptr<uint16_t>(r);
      const cv::Vec3b *itC = color.ptr<cv::Vec3b>(r);
      const float y = lookupY.at<float>(0, r);
      const float *itX = lookupX.ptr<float>();

      for(size_t c = 0; c < (size_t)depth.cols; ++c, ++itP, ++itD, ++itC, ++itX)
      {
        register const float depthValue = *itD / 1000.0f;
        // Check for invalid measurements
        if(isnan(depthValue) || depthValue <= 0.001)
        {
          // not valid
          itP->x = itP->y = itP->z = badPoint;
          itP->rgba = 0;
          continue;
        }
        itP->z = depthValue;
        itP->x = *itX * depthValue;
        itP->y = y * depthValue;
        itP->b = itC->val[0];
        itP->g = itC->val[1];
        itP->r = itC->val[2];
        itP->a = 255;
      }
    }
  }

  void saveCloudAndImages(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud, const cv::Mat &color, const cv::Mat &depth, const cv::Mat &depthColored)
  {
    oss.str("");
    oss << "./" << std::setfill('0') << std::setw(4) << frame;
    const std::string baseName = oss.str();
    const std::string cloudName = baseName + "_cloud.pcd";
    const std::string colorName = baseName + "_color.jpg";
    const std::string depthName = baseName + "_depth.png";
    const std::string depthColoredName = baseName + "_depth_colored.png";

    std::cout << "saving cloud: " << cloudName << std::endl;
    writer.writeBinary(cloudName, *cloud);
    std::cout << "saving color: " << colorName << std::endl;
    cv::imwrite(colorName, color, params);
    std::cout << "saving depth: " << depthName << std::endl;
    cv::imwrite(depthName, depth, params);
    std::cout << "saving depth: " << depthColoredName << std::endl;
    cv::imwrite(depthColoredName, depthColored, params);
    std::cout << "saving complete!" << std::endl;
    ++frame;
  }

  void createLookup(size_t width, size_t height)
  {
    const float fx = 1.0f / cameraMatrixColor.at<double>(0, 0);
    const float fy = 1.0f / cameraMatrixColor.at<double>(1, 1);
    const float cx = cameraMatrixColor.at<double>(0, 2);
    const float cy = cameraMatrixColor.at<double>(1, 2);   
    float *it;

    lookupY = cv::Mat(1, height, CV_32F);
    it = lookupY.ptr<float>();
    for(size_t r = 0; r < height; ++r, ++it)
    {
      *it = (r - cy) * fy;
    }

    lookupX = cv::Mat(1, width, CV_32F);
    it = lookupX.ptr<float>();
    for(size_t c = 0; c < width; ++c, ++it)
    {
      *it = (c - cx) * fx;
    }
  }
};

void help(const std::string &path)
{
  std::cout << path << " [options]" << std::endl
            << "         name: 'any string' equals to the kinect2_bridge topic base name" << std::endl
            << "         mode: 'qhd', 'hd', 'sd' or 'ir'" << std::endl
            << "         visualization: 'image', 'cloud' or 'both'" << std::endl
            << "         options:" << std::endl
            << "         'compressed' use compressed instead of raw topics" << std::endl
            << "         'approx' use approximate time synchronization" << std::endl;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "kinect2_viewer", ros::init_options::AnonymousName);

  if(!ros::ok())
  {
    return 0;
  }

  //load cpu cascades
  face_cascade.load( face_cascade_name );
  eyes_cascade.load( eyes_cascade_name );

  //load gpu cascades
  face_cascade_gpu.load( face_cascade_name );
  eyes_cascade_gpu.load( eyes_cascade_name );

  int gpuCnt = getCudaEnabledDeviceCount();   // gpuCnt >0 if CUDA device detected
  if(gpuCnt==0)
  {
    return 0;                       // no CUDA device found, quit
  } 

  setIdentity(KF.measurementMatrix);
  setIdentity(KF.processNoiseCov, Scalar::all(Qt));
  setIdentity(KF.measurementNoiseCov, Scalar::all(Rt));
  setIdentity(KF.errorCovPost, Scalar::all(1));

  KF.statePost.at<float>(0) = 730;          //initialize kalman posteriori

//second kalman
  setIdentity(KF2.measurementMatrix);
  setIdentity(KF2.processNoiseCov, Scalar::all(Qt));
  setIdentity(KF2.measurementNoiseCov, Scalar::all(Rt2));
  setIdentity(KF2.errorCovPost, Scalar::all(1));

  KF2.statePost.at<float>(0) = 730;          //initialize kalman posteriori

  std::string ns = K2_DEFAULT_NS;
  std::string topicColor = K2_TOPIC_QHD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
  std::string topicDepth = K2_TOPIC_QHD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
  bool useExact = true;
  bool useCompressed = false;
  Receiver::Mode mode = Receiver::CLOUD;

  for(size_t i = 1; i < (size_t)argc; ++i)
  {
    std::string param(argv[i]);

    if(param == "-h" || param == "--help" || param == "-?" || param == "--?")
    {
      help(argv[0]);
      ros::shutdown();
      return 0;
    }
    else if(param == "qhd")
    {
      topicColor = K2_TOPIC_QHD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
      topicDepth = K2_TOPIC_QHD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
    }
    
    else if(param == "hd")
    {
      topicColor = K2_TOPIC_HD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
      topicDepth = K2_TOPIC_HD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
    }
    else if(param == "ir")
    {
      topicColor = K2_TOPIC_SD K2_TOPIC_IMAGE_IR K2_TOPIC_IMAGE_RECT;
      topicDepth = K2_TOPIC_SD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;      
    }
    else if(param == "sd")
    {
      topicColor = K2_TOPIC_SD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
      topicDepth = K2_TOPIC_SD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
    }
    else if(param == "approx")
    {
      useExact = false;
    }

    else if(param == "compressed")
    {
      useCompressed = true;
    }
    else if(param == "image")
    {
      mode = Receiver::IMAGE;
    }
    else if(param == "cloud")
    {
      mode = Receiver::CLOUD;
    }
    else if(param == "both")
    {
      mode = Receiver::BOTH;
    }
    else
    {
      ns = param;
    }
  }

  topicColor = "/" + ns + topicColor;
  topicDepth = "/" + ns + topicDepth;
  std::cout << "topic color: " << topicColor << std::endl;
  std::cout << "topic depth: " << topicDepth << std::endl;

  Receiver receiver(topicColor, topicDepth, useExact, useCompressed);

  std::cout << "starting receiver..." << std::endl;
  receiver.run(mode);
  ros::shutdown();
  return 0;
}