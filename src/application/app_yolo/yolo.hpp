#ifndef YOLO_HPP
#define YOLO_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>
#include <common/object_detector.hpp>

/**
 * @brief 发挥极致的性能体验
 * 支持YoloX和YoloV5
 */
namespace Yolo{

    using namespace std;
    using namespace ObjectDetector;

    enum class Type : int{
        V5 = 0,
        X  = 1
    };

    enum class NMSMethod : int{
        CPU = 0,         // General, for estimate mAP
        FastGPU = 1      // Fast NMS with a small loss of accuracy in corner cases
    };

    enum class ImageType : int{
        CVMat  = 0,
        GPUYUV = 1    // nv12
    };

    struct Image{
        ImageType type = ImageType::CVMat;
        cv::Mat cvmat;

        // GPU YUV image
        TRT::CUStream stream = nullptr;
        uint8_t* device_data = nullptr;
        int width = 0, height = 0;
        int device_id = 0;

        Image() = default;
        Image(const cv::Mat& cvmat):cvmat(cvmat), type(ImageType::CVMat){}
        Image(uint8_t* yuvdata_device, int width, int height, int device_id, TRT::CUStream stream)
        :device_data(yuvdata_device), width(width), height(height), device_id(device_id), stream(stream), type(ImageType::GPUYUV){}

        int get_width() const{return type == ImageType::CVMat ? cvmat.cols : width;}
        int get_height() const{return type == ImageType::CVMat ? cvmat.rows : height;}
        cv::Size get_size() const{return cv::Size(get_width(), get_height());}
        bool empty() const{return type == ImageType::CVMat ? cvmat.empty() : (device_data == nullptr || width < 1 || height < 1);}
    };

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, Type type, int ibatch);

    class Infer{
    public:
        virtual shared_future<BoxArray> commit(const Image& image) = 0;
        virtual vector<shared_future<BoxArray>> commits(const vector<Image>& images) = 0;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, Type type, int gpuid,
        float confidence_threshold=0.25f, float nms_threshold=0.5f,
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024,
        bool use_multi_preprocess_stream = false
    );
    const char* type_name(Type type);

}; // namespace Yolo

#endif // YOLO_HPP