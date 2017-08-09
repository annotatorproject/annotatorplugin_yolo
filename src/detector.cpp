#include "detector.h"
#include <opencv2/core.hpp>

Detector::Detector(std::string cfg_file, std::string weights_file,
                   float threshold)
    : cfg(cfg_file), weights(weights_file), threshold(threshold) {}

void Detector::init() {
  char cfgfile[cfg.size() + 1];
  std::copy(cfg.begin(), cfg.end(), cfgfile);
  char weightfile[weights.size() + 1];
  std::copy(weights.begin(), weights.end(), weightfile);
  net = parse_network_cfg(cfgfile);

  load_weights(&net, weightfile);

  set_batch_network(&net, 1);
}

std::vector<std::vector<float>> Detector::detect(cv::Mat img) {
  std::vector<std::vector<float>> detections;
  image im = matToImage(img);
  image sized = letterbox_image(im, net.w, net.h);
  layer l = net.layers[net.n - 1];

  box *boxes = (box *)calloc(l.w * l.h * l.n, sizeof(box));
  float **probs = (float **)calloc(l.w * l.h * l.n, sizeof(float *));
  for (int j = 0; j < l.w * l.h * l.n; ++j)
    probs[j] = (float *)calloc(l.classes + 1, sizeof(float *));
  float **masks = 0;
  if (l.coords > 4) {
    masks = (float **)calloc(l.w * l.h * l.n, sizeof(float *));
    for (int j = 0; j < l.w * l.h * l.n; ++j)
      masks[j] = (float *)calloc(l.coords - 4, sizeof(float *));
  }
  float *X = sized.data;
  network_predict(net, X);
  get_region_boxes(l, im.w, im.h, net.w, net.h, threshold, probs, boxes, masks,
                   0, 0, 0.5, 1);
  int num = l.w * l.h * l.n;
  for (int i = 0; i < num; ++i) {
    int _class = max_index(probs[i], l.classes);
    float prob = probs[i][_class];
    if (prob > threshold) {
      box b = boxes[i];
      float w = b.w * im.w;
      float h = b.h * im.h;
      float x = b.x * im.w;
      x -= w / 2.0f;
      float y = b.y * im.h;
      y -= h / 2.0f;
      std::vector<float> detection;
      detection.push_back(_class);
      detection.push_back(prob);
      detection.push_back(x);
      detection.push_back(y);
      detection.push_back(w);
      detection.push_back(h);
      detections.push_back(detection);
    }
  }
  free_image(im);
  free_image(sized);
  free(boxes);
  free_ptrs((void **)probs, l.w * l.h * l.n);

  return detections;
}

// source: https://github.com/frankzhangrui/Darknet-Yolo/blob/master/src/image.c
image Detector::matToImage(cv::Mat img) {
  IplImage *src = new IplImage(img);
  unsigned char *data = (unsigned char *)src->imageData;
  int h = src->height;
  int w = src->width;
  int c = src->nChannels;
  int step = src->widthStep;
  image out = make_image(w, h, c);
  int i, j, k, count = 0;

  for (k = 0; k < c; ++k) {
    for (i = 0; i < h; ++i) {
      for (j = 0; j < w; ++j) {
        out.data[count++] = data[i * step + j * c + k] / 255.;
      }
    }
  }
  return out;
}
