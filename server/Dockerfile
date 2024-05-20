FROM ubuntu AS build

# Установите необходимые зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    autoconf \
    libtool \
    pkg-config  \
    g++ \
    cmake \
    wget \
    libboost-system-dev \
    libboost-thread-dev \
    libpthread-stubs0-dev \
    && rm -rf /var/lib/apt/lists/*

# Установите рабочую директорию
WORKDIR /src

# Скопируйте файлы проекта в контейнер
COPY CMakeLists.txt main.cpp WebServer.cpp WebServer.h ImageClassification.cpp ImageClassificaion.h ./

# Установите переменную окружения для CMAKE_PREFIX_PATH
#ENV CMAKE_PREFIX_PATH=/usr/local/libtorch

# Создайте директорию для сборки
RUN g++ -o main main.cpp WebServer.cpp ImageClassification.cpp ImageClassificaion.h -lboost_system -lboost_thread -lpthread

#RUN chmod +x server

RUN cmake . && cmake

EXPOSE 8001

# Укажите команду по умолчанию для запуска приложения
CMD ["./main"]

#FROM ubuntu:latest

#WORKDIR /opt/server

#COPY --from=build /src/server_demo ./
