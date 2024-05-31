#include <iostream>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind/bind.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "WebServer.h"
#include "ImageClassification.h"

using namespace boost::asio;
typedef boost::shared_ptr<ip::tcp::socket> socket_ptr;

   ThreadPool::ThreadPool(size_t num_threads) : boost::asio::thread_pool(num_threads) , num_threads(2), active_tasks(0)
   {
     for (size_t i = 0; i < num_threads; ++i) {
       threads_.emplace_back([this] {
         while (true) {
           std::function<void()> task;
           {
             std::unique_lock<std::mutex> lock(mutex_);
             condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
             if (stop_ && tasks_.empty()) {
               return;
             }
             task = std::move(tasks_.front());
             tasks_.pop();
             active_tasks++;
             sleep(5);
           }
           task();
           {
             std::lock_guard<std::mutex> lock(mutex_);
             --active_tasks;
             condition_.notify_all();
           }

         }
       });
     }
   }

   void ThreadPool::sleep(int duration)
   {
     std::this_thread::sleep_for(std::chrono::seconds (duration));
   }

   template<class F>
   void ThreadPool::enqueue(F&& f) {
     {
       std::unique_lock<std::mutex> lock(mutex_);
       tasks_.emplace(std::forward<F>(f));
     }
     condition_.notify_one();
   }

   ThreadPool::~ThreadPool() {
     {
       std::unique_lock<std::mutex> lock(mutex_);
       stop_ = true;
     }
     condition_.notify_all();
     for (auto& thread : threads_) {
       thread.join();
     }
   }

   int ThreadPool::getQueueSize()
   {
     return this->tasks_.size();
   }

   int ThreadPool::getSize()
   {
     return this->num_threads;
   }

   size_t ThreadPool::get_active_tasks()
   {
     std::lock_guard<std::mutex> lock(mutex_);
     return active_tasks;
   }

   HTTPServer::HTTPServer(io_service& io_service, short port)
     : acceptor_(io_service, ip::tcp::endpoint(ip::tcp::v4(), port)), io_service_(io_service), pool(2)
   {
     start_accept();
   }

   // business logic function
   std::string HTTPServer::getLogic(std::string& str)
   {
     str = str.substr(str.find('=')+1, str.size()-1);
     str = str.substr(0,str.find_first_of(' '));
     return "HTTP/1.1 200 OK\r\nContent-Length: "+std::to_string(str.size())+"\r\n\r\n"+str;
   }
   void HTTPServer::start_accept() {
     socket_ptr socket(new ip::tcp::socket(io_service_));
     acceptor_.async_accept(*socket,
                            [this, socket](auto && PH1) { handle_accept(socket, std::forward<decltype(PH1)>(PH1)); });
   }

   void HTTPServer::handle_accept(const socket_ptr socket, const boost::system::error_code& error) {
     if (!error) {
       if(this->pool.get_active_tasks() >= this->pool.getSize()) {
         //write(*socket,buffer("HTTP/1.1 400 Bad Request\r\n\r\n"));
         write(*socket, boost::asio::buffer("HTTP/1.1 503 Service Unavailable\r\n\r\n"));
         std::cerr << "Error: All threads are busy. Returning 503 Service Unavailable.\n";
       }
       else pool.enqueue([this, socket] { handle_request(socket); });
     }
     start_accept();
   }

   void HTTPServer::handle_request(const socket_ptr socket) {
     const int num_classes = 525;
     try {
       char data[1024];
       size_t length = socket->read_some(buffer(data));
       std::string request(data, length);
       std::cout << request;
       ImageClassifier classifier("path/to/model/weights", num_classes);

       // get image from our device
       // Вызываем метод predict, передав путь к тестовому изображению
       std::string predictedClass = classifier.predict(request);

       // get Logic and avoid HTTP/0.9 while curl call
       std::string response = getLogic(request);
       std::cout << response;
       write(*socket, buffer(response));
     } catch (std::exception& e) {
       std::cerr << "Exception in thread: " << e.what() << '\n';
     }
   }

