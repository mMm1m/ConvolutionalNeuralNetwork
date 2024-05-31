#ifndef IMAGE_CLASSIFICATION_WEBSERVER_H
#define IMAGE_CLASSIFICATION_WEBSERVER_H

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
using namespace boost::asio;
typedef boost::shared_ptr<ip::tcp::socket> socket_ptr;

class ThreadPool : boost::asio::thread_pool {
  public:

   ThreadPool(size_t num_threads);

   template<class F>
   void enqueue(F&& f);

   ~ThreadPool();

   int getQueueSize();

   int getSize();

   size_t get_active_tasks();

   void sleep(int duration);

  private:
   std::vector<std::thread> threads_;
   std::queue<std::function<void()>> tasks_;
   std::mutex mutex_;
   std::condition_variable condition_;
   bool stop_ = false;
   int num_threads;
   size_t active_tasks;
};

class HTTPServer {
  public:
   HTTPServer(io_service& io_service, short port);

  private:
   // business logic function
   std::string getLogic(std::string& str);
   void start_accept();

   void handle_accept(socket_ptr socket, const boost::system::error_code& error);

   void handle_request(socket_ptr socket);

   ip::tcp::acceptor acceptor_;
   io_service& io_service_;
   //boost::asio::thread_pool pool;
   ThreadPool pool;
};

#endif //IMAGE_CLASSIFICATION_WEBSERVER_H
