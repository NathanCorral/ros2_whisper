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
  std::vector<whisper_token> p(std::vector<std::string> & texts, std::vector<float> & probs);

  whisper_context *ctx;
  whisper_full_params wparams;
  whisper_context_params cparams;
};
} // end of namespace whisper
#endif // WHISPER_UTIL__WHISPER_HPP_
