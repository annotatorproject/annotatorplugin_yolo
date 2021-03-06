#ifndef YOLO_H
#define YOLO_H

#include <annotator/plugins/plugin.h>
#include "detector.h"
#include "widget.h"

#include <memory>
#include <thread>
#include <vector>

#include <QtCore/QObject>
#include <QtCore/QtPlugin>
#include <QtGui/QIcon>
#include <opencv2/core/mat.hpp>

using std::shared_ptr;
using namespace AnnotatorLib;

namespace AnnotatorLib {
class Session;
}

namespace Annotator {
namespace Plugins {

class YOLO : public Plugin {
  Q_OBJECT
  Q_PLUGIN_METADATA(IID "annotator.yolo" FILE "yolo.json")
  Q_INTERFACES(Annotator::Plugin)

 public:
  YOLO();
  ~YOLO();
  QString getName() override;
  QWidget *getWidget() override;
  bool setFrame(shared_ptr<Frame> frame, cv::Mat image) override;
  void setObject(shared_ptr<Object> object) override;
  shared_ptr<Object> getObject() const override;
  void setLastAnnotation(shared_ptr<Annotation>) override;
  std::vector<shared_ptr<Commands::Command>> getCommands() override;

  void setPrototxt(std::string file);
  void setModel(std::string file);
  void setLabelmap(std::string file);
  void setConfidenceThreshold(float threshold);

  virtual bool requiresObject() const override { return false; }

 protected:
  int epoch = 1;
  int width = 300;
  int height = 300;
  float mean_r = 123.f;
  float mean_g = 117.f;
  float mean_b = 104.f;
  int device_type = 1;
  int device_id = 0;

  bool modelLoaded = false;
  std::string prototxt_file;
  std::string model_file;
  std::string labelmap_file;
  float confidence_threshold = 0.01f;
  std::vector<std::string> labels;

  Detector *detector = nullptr;

  void initDetector();
  std::shared_ptr<AnnotatorLib::Class> getClass(int label);

  QPixmap getImgCrop(shared_ptr<AnnotatorLib::Annotation> annotation,
                     int size) const;
  cv::Mat getImg(shared_ptr<AnnotatorLib::Annotation> annotation) const;

  cv::Mat frameImg;
  shared_ptr<Annotation> lastAnnotation = nullptr;
  shared_ptr<Object> object = nullptr;

  Widget widget;
  std::thread trainThread;

  shared_ptr<Frame> frame = nullptr;
  shared_ptr<Frame> lastFrame = nullptr;
};
}
}

#endif  // YOLO_H
