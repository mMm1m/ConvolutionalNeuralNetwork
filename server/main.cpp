#include <iostream>
#include "WebServer.h"

int main()
{
  // Создаем экземпляр класса ImageClassifier, передав путь к весам модели
  // в нашем случае 525 классов;
  try
  {
    io_service service;
    HTTPServer server(service , 8001);
    service.run();
  }
  catch(std::exception& e)
  {
    std::cerr << e.what();
  }
  return 0;
}