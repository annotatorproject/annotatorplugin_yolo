#ifndef DETECTOR_H
#define DETECTOR_H

#include <string>

#include <opencv2/core/mat.hpp>

extern "C" {
#include <darknet.h>
}

class Detector {
 public:
  Detector(std::string cfg_file, std::string weights_file,
           float threshold = 0.24);
  void init();
  std::vector<std::vector<float>> detect(cv::Mat img);

 protected:
  image matToImage(cv::Mat img);

  network net;

  std::string cfg;
  std::string weights;
  float threshold;
};

#endif  // DETECTOR_H
