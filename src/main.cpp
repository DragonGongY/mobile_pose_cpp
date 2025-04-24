#include "mobile_pose.hpp"

int main() {
    
    
    MobilePose mobile_pose;

    // Open video
    cv::VideoCapture cap("../flip2.mp4");
    while (cap.isOpened()) {
        cv::Mat frame;
        bool ret = cap.read(frame);
        if (!ret) break;
        // Run inference
        std::vector<cv::Point> humans = mobile_pose.inference(frame);
        
        // Draw keypoints
        for (const auto& point : humans) {
            cv::circle(frame, point, 5, cv::Scalar(0, 255, 0), -1);
        }
        
        // Display frame
        cv::imshow("MobilePose Demo", frame);
        
        if (cv::waitKey(1) & 0xFF == 'q') break;
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    
    return 0;
}