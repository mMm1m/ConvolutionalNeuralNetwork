#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

class ImageClassifier {
public:
    ImageClassifier(const std::string& modelWeightsPath, const int& num_classes) {
        std::string str = "class_";
        // Присваиваем имена классам
        for(int i = 0; i < num_classes; ++i) class_names.push_back(str + std::to_string(num_classes));
        // Загружаем предварительно обученную модель ResNet-18
        model = torch::vision::models::ResNet::resnet18();

        // Меняем последний слой классификатора
        int num_ftrs = model.fc->in_features;
        model.fc = torch::nn::Linear(num_ftrs, 2);

        // Загружаем веса модели
        torch::load(model, modelWeightsPath);

        // Устанавливаем в режим оценки
        model->eval();

        // Переводим модель в режим работы с CUDA, если доступно
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
            model->to(device);
        }
    }

    std::string predict(const std::string& imagePath) {
        // Загружаем изображение с помощью OpenCV
        cv::Mat image = cv::imread(imagePath);
        torch::Tensor tensor = vision::transforms::functional::MatToTensor(image);

        // Добавляем размерность пакета
        tensor = torch::unsqueeze(tensor, 0);

        // Переводим тензор в режим работы с CUDA, если возможно
        if (torch::cuda::is_available()) {
            tensor = tensor.to(device);
        }

        // Получаем выход модели
        torch::Tensor output = model->forward(tensor);

        // Получаем индекс класса с наибольшей вероятностью
        torch::Tensor probabilities = torch::softmax(output, /*dim=*/1);
        torch::Tensor topPredicted = torch::argmax(probabilities, /*dim=*/1);
        int predictedClassIndex = topPredicted.item<int>();

        // Получаем предполагаемый класс
        std::string predictedClass = class_names[predictedClassIndex];

        return predictedClass;
    }

private:
    torch::Device device;
    torch::nn::Sequential model;
    std::vector<std::string> class_names;
};

int main() {
    // Создаем экземпляр класса ImageClassifier, передав путь к весам модели
    // в нашем случае 525 классов;
    int num_classes = 525;
    ImageClassifier classifier("path/to/model/weights", num_classes);

    // Вызываем метод predict, передав путь к тестовому изображению
    std::string predictedClass = classifier.predict("path/to/image");

    // Выводим результат классификации
    std::cout << "Predicted class: " << predictedClass << std::endl;

    return 0;
}
