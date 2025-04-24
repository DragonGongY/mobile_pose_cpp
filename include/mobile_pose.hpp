#pragma once
#include <onnxruntime_cxx_api.h>

#include <functional>
#include <opencv2/opencv.hpp>
#include <vector>

struct RescaleOutput {
  cv::Mat image;
  std::function<std::vector<cv::Point2f>(const std::vector<float>&)> pose_fun;
};

class MobilePose {
 public:
  MobilePose(const std::string& model_path = "../models/mobilenetv2_pose-sim.onnx");
  ~MobilePose();

  RescaleOutput rescale(const cv::Mat& image, const cv::Size& output_size);
  std::vector<cv::Point> inference(const cv::Mat& in_img);
  cv::Mat crop_camera(const cv::Mat& image, float ratio);

 private:
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "MobilePose"};
  Ort::Session* session;
  Ort::AllocatorWithDefaultOptions allocator;
};