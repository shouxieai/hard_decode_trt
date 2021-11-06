
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_yolo/yolo.hpp"
#include <ffhdd/ffmpeg_demuxer.hpp>
#include <ffhdd/cuvid_decoder.hpp>
#include <ffhdd/nalu.hpp>

using namespace std;

static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

bool requires(const char* name);

static shared_ptr<Yolo::Infer> get_yolo(Yolo::Type type, TRT::Mode mode, const string& model, int device_id){

    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(device_id);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);
            Yolo::image_to_tensor(image, tensor, type, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test %s %s %s ==================================", Yolo::type_name(type), mode_name, name);

    if(not requires(name))
        return nullptr;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;
    
    if(not iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file,                 // save to
            {},
            int8process,
            "inference"
        );
    }

    return Yolo::create_infer(
        model_file,                 // engine file
        type,                       // yolo type, Yolo::Type::V5 / Yolo::Type::X
        device_id,                  // gpu id
        0.25f,                      // confidence threshold
        0.45f,                      // nms threshold
        Yolo::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
        1024,                       // max objects
        false                       // preprocess use multi stream
    );
}

void render_to_images(vector<shared_future<Yolo::BoxArray>>& objs_array, const string& name){

    iLogger::rmtree(name);
    iLogger::mkdir(name);

    cv::VideoCapture capture("exp/fall_video.mp4");
    cv::Mat image;

    if(!capture.isOpened()){
        INFOE("Open video failed.");
        return;
    }

    int iframe = 0;
    while(capture.read(image) && iframe < objs_array.size()){
        auto objs = objs_array[iframe].get();
        for(auto& obj : objs){
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(obj.class_label);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

            auto name    = cocolabels[obj.class_label];
            auto caption = iLogger::format("%s %.2f", name, obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }

        cv::imwrite(iLogger::format("%s/%03d.jpg", name.c_str(), iframe), image);
        iframe++;
    }
}

static void test_hard_decode(){

    auto name = "hard";
    int yolo_device_id = 0;
    auto yolo = get_yolo(Yolo::Type::V5, TRT::Mode::FP32, "yolov5s", yolo_device_id);
    if(yolo == nullptr){
        INFOE("Yolo create failed");
        return;
    }

    auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer("exp/fall_video.mp4");
    if(demuxer == nullptr){
        INFOE("demuxer create failed");
        return;
    }

    int decoder_device_id = 0;
    auto decoder = FFHDDecoder::create_cuvid_decoder(
        true, FFHDDecoder::ffmpeg2NvCodecId(demuxer->get_video_codec()), -1, decoder_device_id
    );

    if(decoder == nullptr){
        INFOE("decoder create failed");
        return;
    }

    uint8_t* packet_data = nullptr;
    int packet_size = 0;
    int64_t pts = 0;

    /* 这个是头，具有一些信息，但是没有图像数据 */
    demuxer->get_extra_data(&packet_data, &packet_size);
    decoder->decode(packet_data, packet_size);

    // warm up
    for(int i = 0; i < 10; ++i)
        yolo->commit(cv::Mat(640, 640, CV_8UC3)).get();
    
    vector<shared_future<Yolo::BoxArray>> all_boxes;
    auto tic = iLogger::timestamp_now_float();
    do{
        demuxer->demux(&packet_data, &packet_size, &pts);
        int ndecoded_frame = decoder->decode(packet_data, packet_size, pts);
        for(int i = 0; i < ndecoded_frame; ++i){
            unsigned int frame_index = 0;

            Yolo::Image image(
                decoder->get_frame(&pts, &frame_index), 
                decoder->get_width(), decoder->get_height(), 
                decoder_device_id,
                decoder->get_stream()
            );
            all_boxes.emplace_back(yolo->commit(image));
        }
    }while(packet_size > 0);

    auto toc = iLogger::timestamp_now_float();
    INFO("%s decode and inference time: %.2f ms", name, toc - tic);

    render_to_images(all_boxes, name);
}

static void test_soft_decode(){

    auto name = "soft";
    int yolo_device_id = 0;
    auto yolo = get_yolo(Yolo::Type::V5, TRT::Mode::FP32, "yolov5s", yolo_device_id);
    if(yolo == nullptr){
        INFOE("Yolo create failed");
        return;
    }

    cv::VideoCapture capture("exp/fall_video.mp4");
    cv::Mat image;

    if(!capture.isOpened()){
        INFOE("Open video failed.");
        return;
    }

    // warm up
    for(int i = 0; i < 10; ++i)
        yolo->commit(cv::Mat(640, 640, CV_8UC3)).get();
    
    vector<shared_future<Yolo::BoxArray>> all_boxes;
    auto tic = iLogger::timestamp_now_float();
    while(capture.read(image)){
        all_boxes.emplace_back(yolo->commit(image));
    }

    auto toc = iLogger::timestamp_now_float();
    INFO("soft decode and inference time: %.2f ms", toc - tic);

    render_to_images(all_boxes, name);
}

int app_yolo(){

    test_soft_decode();
    test_hard_decode();
    return 0;
}