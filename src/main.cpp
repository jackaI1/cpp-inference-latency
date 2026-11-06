#include "stats.h"
#include <onnxruntime_cxx_api.h>
#include <sys/resource.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
 try {
  if (argc != 7) throw std::runtime_error("usage: benchmark MODEL DATA BATCH WARMUP RUNS OUTPUT.csv");
  const int batch = std::stoi(argv[3]), warmup = std::stoi(argv[4]), runs = std::stoi(argv[5]);
  if (batch <= 0 || warmup < 0 || runs <= 0) throw std::runtime_error("invalid batch, warmup or runs");
  std::ifstream in(argv[2]); if (!in) throw std::runtime_error("cannot open data");
  std::vector<float> images; std::vector<int> labels;
  int label; while (in >> label) { labels.push_back(label); for (int j=0;j<64;++j) { float x; if (!(in >> x)) throw std::runtime_error("incomplete image"); images.push_back(x); } }
  if (labels.empty() || labels.size() % batch) throw std::runtime_error("image count must be nonzero and divisible by batch");
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "benchmark"); Ort::SessionOptions options; options.SetIntraOpNumThreads(1);
  Ort::Session session(env, argv[1], options); Ort::AllocatorWithDefaultOptions allocator;
  auto input_name = session.GetInputNameAllocated(0, allocator); auto output_name = session.GetOutputNameAllocated(0, allocator);
  const char* inputs[] = {input_name.get()}; const char* outputs[] = {output_name.get()};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<double> times; std::vector<float> predictions; int correct = 0;
  for (int r=-warmup; r<runs; ++r) for (size_t offset=0; offset<labels.size(); offset+=batch) {
   std::vector<int64_t> shape{batch,1,8,8};
   auto tensor = Ort::Value::CreateTensor<float>(memory, images.data()+offset*64, batch*64, shape.data(), shape.size());
   auto start=std::chrono::steady_clock::now();
   auto result=session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
   auto end=std::chrono::steady_clock::now();
   if (r >= 0) times.push_back(std::chrono::duration<double,std::milli>(end-start).count());
   if (r == 0) { auto info=result[0].GetTensorTypeAndShapeInfo(); auto dims=info.GetShape();
    if (dims.size()!=2 || dims[0]!=batch || dims[1]!=2) throw std::runtime_error("expected output [batch,2]");
    const float* p=result[0].GetTensorData<float>();
    for(int i=0;i<batch;++i) { predictions.push_back(p[2*i]); predictions.push_back(p[2*i+1]); correct += (p[2*i+1]>p[2*i]) == labels[offset+i]; }
   }
  }
  std::ofstream out(argv[6]); if (!out) throw std::runtime_error("cannot open output");
  out << std::setprecision(9); for(size_t i=0;i<labels.size();++i) out << labels[i] << ',' << predictions[2*i] << ',' << predictions[2*i+1] << '\n';
  struct rusage usage{}; getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
  double peak_mb=usage.ru_maxrss / 1048576.0;
#else
  double peak_mb=usage.ru_maxrss / 1024.0;
#endif
  double total=std::accumulate(times.begin(),times.end(),0.0);
  std::cout << "images=" << labels.size() << " accuracy=" << double(correct)/labels.size() << " p50_ms=" << percentile(times,.5) << " p95_ms=" << percentile(times,.95) << " images_per_second=" << (1000.0*batch*times.size()/total) << " peak_rss_mib=" << peak_mb << '\n';
 } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
