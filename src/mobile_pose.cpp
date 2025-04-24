#include "mobile_pose.hpp"
#include <array>

MobilePose::MobilePose(const std::string& model_path) {
  Ort::SessionOptions session_options;
  session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

  session_options.SetIntraOpNumThreads(1);
  session_options.SetInterOpNumThreads(1);

  session_options.EnableCpuMemArena();

  try {
    session = new Ort::Session(env, model_path.c_str(), session_options);
  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
    throw; 
  }
}

MobilePose::~MobilePose() { delete session; }

RescaleOutput MobilePose::rescale(const cv::Mat& image,
                                  const cv::Size& output_size) {
  // Normalize image by dividing by 256.0
  cv::Mat image_;
  image.convertTo(image_, CV_32F, 1.0 / 256.0);

  // Get dimensions
  int h = image_.rows;
  int w = image_.cols;

  // Calculate scale to preserve aspect ratio
  float im_scale = std::min(float(output_size.height) / float(h),
                            float(output_size.width) / float(w));
  int new_h = std::ceil(h * im_scale);
  int new_w = int(w * im_scale);

  // Resize image
  cv::Mat resized;
  cv::resize(image_, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

  // Calculate padding
  int left_pad = int((output_size.width - new_w) / 2.0);
  int top_pad = int((output_size.height - new_h) / 2.0);

  // Define mean values
  std::vector<float> mean = {0.485f, 0.456f, 0.406f};

  // Create padded output image
  cv::Mat padded_output(output_size, CV_32FC3);

  // For each channel, pad with mean value
  std::vector<cv::Mat> channels(3);
  cv::split(resized, channels);

  std::vector<cv::Mat> padded_channels(3);
  for (int c = 0; c < 3; c++) {
    cv::Mat padded_channel(output_size, CV_32F, cv::Scalar(mean[c]));
    channels[c].copyTo(
        padded_channel(cv::Rect(left_pad, top_pad, new_w, new_h)));
    padded_channels[c] = padded_channel;
  }

  cv::merge(padded_channels, padded_output);

  // Create pose transformation function
  auto pose_fun =
      [=](const std::vector<float>& keypoints) -> std::vector<cv::Point2f> {
    std::vector<cv::Point2f> transformed_points;
    size_t num_keypoints = keypoints.size() / 2;

    for (size_t i = 0; i < num_keypoints; i++) {
      float x = keypoints[i * 2];
      float y = keypoints[i * 2 + 1];

      // Add [1.0, 1.0]
      x += 1.0;
      y += 1.0;

      // Divide by 2.0
      x /= 2.0;
      y /= 2.0;

      // Multiply by output_size
      x *= output_size.width;
      y *= output_size.height;

      // Subtract [left_pad, top_pad]
      x -= left_pad;
      y -= top_pad;

      // Multiply by 1.0/[new_w, new_h]
      x /= new_w;
      y /= new_h;

      // Multiply by [w, h]
      x *= w;
      y *= h;

      transformed_points.push_back(cv::Point2f(x, y));
    }

    return transformed_points;
  };

  RescaleOutput output;
  output.image = padded_output;
  output.pose_fun = pose_fun;

  return output;
}

std::vector<cv::Point2i> MobilePose::inference(const cv::Mat& in_img) {
  try {
    cv::Mat cropped = crop_camera(in_img, 0.15);
    // Rescale the image
    RescaleOutput rescale_out = rescale(cropped, cv::Size(224, 224));
    cv::Mat image = rescale_out.image;

    // Normalize image with mean and std
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};

    // Apply normalization
    cv::Mat normalized_image = image.clone();
    for (int h = 0; h < normalized_image.rows; h++) {
      for (int w = 0; w < normalized_image.cols; w++) {
        cv::Vec3f& pixel = normalized_image.at<cv::Vec3f>(h, w);
        for (int c = 0; c < 3; c++) {
          pixel[c] = (pixel[c] - mean[c]) / std[c];
        }
      }
    }

    // Convert to NCHW format (transpose)
    std::vector<cv::Mat> channels(3);
    cv::split(normalized_image, channels);

    // Create input tensor data
    const size_t input_tensor_size = 1 * 3 * 224 * 224;
    std::vector<float> input_tensor_values(input_tensor_size);

    // Copy data to input tensor values
    for (int c = 0; c < 3; c++) {
      for (int h = 0; h < 224; h++) {
        for (int w = 0; w < 224; w++) {
          input_tensor_values[c * 224 * 224 + h * 224 + w] =
              channels[c].at<float>(h, w);
        }
      }
    }

    // Create input tensor
    std::vector<int64_t> input_dims = {1, 3, 224, 224};
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_dims.data(), input_dims.size());

    // Define input and output names
    Ort::AllocatorWithDefaultOptions allocator;

    const char* input_name = "input";
    const char* output_name = "output";

    std::vector<const char*> input_names = {input_name};
    std::vector<const char*> output_names = {output_name};

    // 使用ONNX Runtime进行推理
    std::vector<Ort::Value> outputs =
        session->Run(Ort::RunOptions{nullptr}, input_names.data(),
                     &input_tensor, 1, output_names.data(), 1);

    if (outputs.empty()) {
      throw std::runtime_error("No output from model");
    }

    // Get output data
    const float* output_data = outputs[0].GetTensorData<float>();

    // Get output dimensions
    auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int output_size = 1;
    for (auto dim : output_shape) {
      output_size *= dim;
    }

    // Extract keypoints
    std::vector<float> keypoints(output_data, output_data + output_size);

    // Transform keypoints
    std::vector<cv::Point2f> transformed_points =
        rescale_out.pose_fun(keypoints);

    // Convert to integer points
    std::vector<cv::Point> result;
    for (const auto& pt : transformed_points) {
      result.push_back(
          cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
    }

    return result;
  } catch (const std::exception& e) {
    std::cerr << "Error during inference: " << e.what() << std::endl;
    return {};
  }
}

cv::Mat MobilePose::crop_camera(const cv::Mat& image, float ratio) {
  int height = image.rows;
  int width = image.cols;
  float mid_width = width / 2.0;
  float width_20 = width * ratio;

  int left = std::max(0, int(mid_width - width_20));
  int top = 0;
  int crop_width = std::min(int(width_20 * 2), width - left);
  int crop_height = height;

  cv::Rect roi(left, top, crop_width, crop_height);
  return image(roi).clone();
}