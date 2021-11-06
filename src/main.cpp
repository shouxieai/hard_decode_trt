
#include <stdio.h>
#include <string.h>
#include <common/ilogger.hpp>
#include <functional>

int app_demuxer();
int app_hard_decode();
int app_yolo();

int main(int argc, char** argv){
    
    const char* method = "yolo";
    if(argc > 1){
        method = argv[1];
    }

    if(strcmp(method, "demuxer") == 0){
        app_demuxer();
    }else if(strcmp(method, "hard_decode") == 0){
        app_hard_decode();
    }else if(strcmp(method, "yolo") == 0){
        app_yolo();
    }else{
        printf("Unknow method: %s\n", method);
        printf(
            "Help: \n"
            "    ./pro method[demuxer]\n"
            "\n"
            "    ./pro yolo\n"
            "    ./pro alphapose\n"
            "    ./pro fall\n"
        );
    }
    return 0;
}
