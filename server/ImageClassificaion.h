#include <iostream>
//#include <torch/torch.h>

#ifndef IMAGE_CLASSIFICATION_IMAGECLASSIFICAION_H
#define IMAGE_CLASSIFICATION_IMAGECLASSIFICAION_H

class ImageClassifier {
  public:
   ImageClassifier(const std::string& modelWeightsPath, const int& num_classes);

   std::string predict(const std::string& imagePath);

  //private:
   //torch::Device device;
   //torch::nn::Sequential model;
   //std::vector<std::string> class_names;
};

#endif //IMAGE_CLASSIFICATION_IMAGECLASSIFICAION_H
