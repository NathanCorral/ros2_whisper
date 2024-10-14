#ifndef WHISPER_UTIL__WHISPER_HPP_
#define WHISPER_UTIL__WHISPER_HPP_

#include <numeric>
#include <string>
#include <vector>
#include <cassert>

#include <cstdio> 

#include "whisper.h"

namespace whisper {
class Whisper {
public:
  Whisper();
  Whisper(const std::string &model_path);
  ~Whisper();

  void initialize(const std::string &model_path);
  std::string forward(const std::vector<float> &input);
  std::vector<whisper_token> tokens();
  void p(std::vector<std::string> & texts, std::vector<float> & probs);
  void get_token_data(std::vector<std::string> & texts, 
                              std::vector<int64_t> & t0s,
                              std::vector<int64_t> & t1s);
  void get_segment_data(std::vector<std::string> & texts, 
                              std::vector<int64_t> & t0s,
                              std::vector<int64_t> & t1s);
  void get_segment_and_token_data(std::vector<std::string> & segs, 
                              std::vector<int64_t> & t0s,
                              std::vector<int64_t> & t1s,
                              std::vector<std::vector<std::string>> & seg_tokens,
                              std::vector<std::vector<float>> & seg_token_probs);

  whisper_context *ctx;
  whisper_full_params wparams;
  whisper_context_params cparams;
};
} // end of namespace whisper
#endif // WHISPER_UTIL__WHISPER_HPP_
