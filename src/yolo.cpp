
// source:
// https://github.com/pjreddie/darknet
#include "yolo.h"
#include "widget.h"

#include <AnnotatorLib/Annotation.h>
#include <AnnotatorLib/Commands/NewAnnotation.h>
#include <AnnotatorLib/Commands/UpdateAnnotation.h>
#include <AnnotatorLib/Frame.h>
#include <AnnotatorLib/Object.h>
#include <AnnotatorLib/Session.h>

#include <ctype.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace Annotator::Plugins;

Annotator::Plugins::YOLO::YOLO() { widget.setYOLO(this); }

YOLO::~YOLO() {}

QString YOLO::getName() { return "YOLO"; }

QWidget *YOLO::getWidget() { return &widget; }

bool YOLO::setFrame(shared_ptr<Frame> frame, cv::Mat image) {
  this->lastFrame = this->frame;
  this->frame = frame;
  this->frameImg = image;
  return lastFrame != frame;
}

// first call
void YOLO::setObject(shared_ptr<Object> object) {
  if (object != this->object) {
    this->object = object;
  }
}

shared_ptr<Object> YOLO::getObject() const { return object; }

void YOLO::setLastAnnotation(shared_ptr<Annotation> /*annotation*/) {}

std::vector<shared_ptr<Commands::Command>> YOLO::getCommands() {
  std::vector<shared_ptr<Commands::Command>> commands;
  if (frame == nullptr || lastFrame == nullptr || lastFrame == frame ||
      frameImg.cols < 1)
    return commands;

  try {
    initDetector();
    std::vector<std::vector<float>> detections =
        detector->detect(this->frameImg);

    int label;
    float score;
    int xmin;
    int ymin;
    int width;
    int height;

    for (int i = 0; i < detections.size(); ++i) {
      const std::vector<float> &d = detections[i];
      // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
      score = d[1];
      if (score >= confidence_threshold) {
        label = static_cast<int>(d[0]);
        xmin = static_cast<int>(d[2]);
        ymin = static_cast<int>(d[3]);
        width = static_cast<int>(d[4]);
        height = static_cast<int>(d[5]);

        std::shared_ptr<Class> labelClass = getClass(label);
        if (labelClass && width > 0 && height > 0) {
          unsigned long objectId = AnnotatorLib::Object::genId();
          shared_ptr<Commands::NewAnnotation> nA =
              std::make_shared<Commands::NewAnnotation>(
                  objectId, labelClass, project->getSession(), this->frame,
                  xmin, ymin, width, height, score, false);
          commands.push_back(nA);
        }
      }
    }
  } catch (std::exception &e) {
  }

  return commands;
}

void YOLO::setPrototxt(std::string file) {
  this->prototxt_file = file;
  modelLoaded = false;
}

void YOLO::setModel(std::string file) {
  this->model_file = file;
  modelLoaded = false;
}

void YOLO::setLabelmap(std::string file) {
  this->labelmap_file = file;
  this->labels.clear();
  std::string label;
  std::ifstream ifs(file, std::ios::in);
  while (std::getline(ifs, label)) labels.push_back(std::move(label));
  modelLoaded = false;
}

void YOLO::setConfidenceThreshold(float threshold) {
  this->confidence_threshold = threshold;
}

void YOLO::initDetector() {
  if (modelLoaded) return;
  if (detector) {
    delete detector;
  }
  detector = new Detector(this->prototxt_file, this->model_file,
                          this->confidence_threshold);
  detector->init();
  modelLoaded = true;
}

std::shared_ptr<Class> YOLO::getClass(int label) {
  try {
    std::string className = labels[label];
    return this->project->getSession()->getClass(className);
  } catch (...) {
    return nullptr;
  }
}

QPixmap YOLO::getImgCrop(shared_ptr<AnnotatorLib::Annotation> annotation,
                         int size) const {
  if (annotation == nullptr) return QPixmap();

  cv::Mat cropped = getImg(annotation);

  cropped.convertTo(cropped, CV_8U);
  cv::cvtColor(cropped, cropped, CV_BGR2RGB);

  QImage img((const unsigned char *)(cropped.data), cropped.cols, cropped.rows,
             cropped.step, QImage::Format_RGB888);

  QPixmap pim = QPixmap::fromImage(img);
  pim = pim.scaledToHeight(size);
  return pim;
}

cv::Mat YOLO::getImg(shared_ptr<Annotation> annotation) const {
  cv::Mat tmp = project->getImageSet()->getImage(
      annotation->getFrame()->getFrameNumber());

  float x = std::max(annotation->getX(), 0.f);
  float y = std::max(annotation->getY(), 0.f);
  float w = std::min(annotation->getWidth(), tmp.cols - x);
  float h = std::min(annotation->getHeight(), tmp.rows - y);

  cv::Rect rect(x, y, w, h);
  cv::Mat cropped;
  try {
    tmp(rect).copyTo(cropped);
  } catch (cv::Exception &e) {
    std::cout << e.what();
  }
  return cropped;
}
