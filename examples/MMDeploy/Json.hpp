
#include <json/json.h>

#include <fstream>
#include <iostream>
#include <string>

#include "Common.hpp"

using namespace std;

struct ResizeArgs {
  Interpolation interpolation;
  std::vector<int> shape;
  bool dynamic = true;
};

struct CropArgs {
  std::vector<int> shape;
  std::vector<int> tlbr;
  bool dynamic = true;
};

struct NormArgs {
  std::vector<float> mean;
  std::vector<float> std;
};

struct PadArgs {
  std::vector<int> paddings;
  std::vector<int> shape;
  float pad_val;
  bool dynamic = true;
};

void readOpList(const std::string &filepath, std::vector<std::string> &OpList,
                Format &CvtFormat) {
  Json::Reader reader;
  Json::Value root;

  ifstream in(filepath.c_str(), ios::binary);

  if (!in.is_open()) {
    ELENA_ABORT("Error opening json file");
    return;
  }

  if (reader.parse(in, root)) {
    if (!root.size()) ELENA_ABORT("json file not have OpList type");
    for (unsigned int i = 0; i < root.size(); i++) {
      string type = root[i]["type"].asString();
      OpList.push_back(type);

      auto mem = root[i];

      if (type == "cvtColorBGR")
        CvtFormat = BGR;
      else if (type == "cvtColorGray")
        CvtFormat = GRAY;
      else if (type != "Resize" && type != "CenterCrop" &&
               type != "Normalize" && type != "Pad" && type != "CastFloat" &&
               type != "cvtColorRGB" && type != "HWC2CHW")
        ELENA_ABORT("unrecognized op type");
    }
  } else {
    ELENA_ABORT("json file is emtpy");
  }

  in.close();
}