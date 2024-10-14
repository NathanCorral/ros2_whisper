#include "whisper_util/whisper.hpp"

namespace whisper {
Whisper::Whisper() { wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY); }

Whisper::Whisper(const std::string &model_path) { initialize(model_path); }

Whisper::~Whisper() { whisper_free(ctx); }

void Whisper::initialize(const std::string &model_path) {
  ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
}

std::string Whisper::forward(const std::vector<float> &input) {
  if (whisper_full(ctx, wparams, input.data(), input.size()) != 0) {
    return {};
  }
  std::vector<std::string> segments;
  int n_segments = whisper_full_n_segments(ctx);
  for (int i = 0; i < n_segments; ++i) {
    segments.push_back(whisper_full_get_segment_text(ctx, i));
  }
  return std::accumulate(segments.begin(), segments.end(), std::string());
}

std::vector<whisper_token> Whisper::tokens() {
  std::vector<whisper_token> tokens;
  const int n_segments = whisper_full_n_segments(ctx);
  for (int i = 0; i < n_segments; ++i) {
    const int token_count = whisper_full_n_tokens(ctx, i);
    for (int j = 0; j < token_count; ++j) {
      tokens.push_back(whisper_full_get_token_id(ctx, i, j));
    }
  }
  return tokens;
}

// std::vector<whisper_token> Whisper::p(std::vector<std::string> & texts, std::vector<float> probs) {
//   std::vector<whisper_token> tokens;
  
//   const int n_segments = whisper_full_n_segments(ctx);
//   for (int i = 0; i < n_segments; ++i) {
//     const int token_count = whisper_full_n_tokens(ctx, i);
//     for (int j = 0; j < token_count; ++j) {
//       tokens.push_back(whisper_full_get_token_id(ctx, i, j));
//       const char * text = whisper_full_get_token_text(ctx, i, j);
//       const float  p    = whisper_full_get_token_p   (ctx, i, j);
//       // Convert text to std::string and push back in texts
//       texts.push_back(text);      
//       probs.push_back(p);
//     }
//   }
//   return tokens;
// }

void Whisper::p(std::vector<std::string> & texts, std::vector<float> & probs) { // Pass probs by reference too  
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const int token_count = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < token_count; ++j) {
            const char * text = whisper_full_get_token_text(ctx, i, j);
            const float  p    = whisper_full_get_token_p   (ctx, i, j);
            if (text != nullptr) {  
                texts.push_back(text);
                probs.push_back(p);
            }
        }
    }
    return;
}

void Whisper::get_segment_data(std::vector<std::string> & texts, 
                              std::vector<int64_t> & t0s,
                              std::vector<int64_t> & t1s) { 
  const int n_segments = whisper_full_n_segments(ctx);
  for (int i = 0; i < n_segments; ++i) {
    const char * text = whisper_full_get_segment_text(ctx, i);
    const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
    const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
    texts.push_back(text);
    t0s.push_back(t0);
    t1s.push_back(t1);
  }
  return;
}

void Whisper::get_token_data(std::vector<std::string> & texts, 
                              std::vector<int64_t> & t0s,
                              std::vector<int64_t> & t1s) { 
  const int n_segments = whisper_full_n_segments(ctx);
  for (int i = 0; i < n_segments; ++i) {
    const int token_count = whisper_full_n_tokens(ctx, i);
    for (int j = 0; j < token_count; ++j) {
      const char * token_text = whisper_full_get_token_text(ctx, i, j);
      auto token_data = whisper_full_get_token_data(ctx, i, j);
      texts.push_back(token_text);
      t0s.push_back(token_data.t0);
      t1s.push_back(token_data.t1);
    }
  }
  return;
}


void Whisper::get_segment_and_token_data(std::vector<std::string> & segs, 
                              std::vector<int64_t> & t0s,
                              std::vector<int64_t> & t1s,
                              std::vector<std::vector<std::string>> & seg_tokens,
                              std::vector<std::vector<float>> & seg_token_probs) {
  const int n_segments = whisper_full_n_segments(ctx);
  for (int i = 0; i < n_segments; ++i) {
    // Get segment data
    const char * seg_text = whisper_full_get_segment_text(ctx, i);
    const int64_t seg_t0 = whisper_full_get_segment_t0(ctx, i);
    const int64_t seg_t1 = whisper_full_get_segment_t1(ctx, i);

    // Get token data
    std::vector<float> probs;
    std::vector<std::string> tokens;
    const int token_count = whisper_full_n_tokens(ctx, i);
    for (int j = 0; j < token_count; ++j) {
      const char * token_text = whisper_full_get_token_text(ctx, i, j);
      const float p = whisper_full_get_token_p(ctx, i, j);
      tokens.push_back(token_text);
      probs.push_back(p);
    }

    // Push data
    segs.push_back(seg_text);
    t0s.push_back(seg_t0);
    t1s.push_back(seg_t1);
    seg_tokens.push_back(tokens);
    seg_token_probs.push_back(probs);
  }
}

} // end of namespace whisper
