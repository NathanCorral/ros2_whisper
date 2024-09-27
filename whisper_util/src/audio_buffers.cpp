#include "whisper_util/audio_buffers.hpp"

namespace whisper {


BatchedBuffer::BatchedBuffer(const std::chrono::milliseconds &buffer_capacity,
                             const std::chrono::milliseconds &carry_over_capacity)
    : carry_over_capacity_(time_to_count(carry_over_capacity)), 
      audio_buffer_(time_to_count(buffer_capacity)) {

      };

void BatchedBuffer::enqueue(const std::vector<std::int16_t> &audio) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto &data : audio) {
    audio_buffer_.enqueue(data);
  }
}

void BatchedBuffer::dequeue(std::vector<float> & result) {
  std::lock_guard<std::mutex> lock(mutex_);
  result.reserve(result.size() + audio_buffer_.size());
  for (std::size_t i = 0; i < audio_buffer_.size(); ++i) {
    result.push_back(static_cast<float>(audio_buffer_.dequeue()) /
                     static_cast<float>(std::numeric_limits<std::int16_t>::max()));
  }
}

void BatchedBuffer::dequeue(std::unique_ptr<RingBuffer<float>> & result) {
  std::lock_guard<std::mutex> lock(mutex_);
  // result.reserve(result.size() + audio_buffer_.size());
  for (std::size_t i = 0; i < audio_buffer_.size(); ++i) {
    result->enqueue(static_cast<float>(audio_buffer_.dequeue()) /
                     static_cast<float>(std::numeric_limits<std::int16_t>::max()));
  }
}

void BatchedBuffer::clear_and_carry_over_(std::vector<float> & data) {
  size_t carry_over_samples = std::min(data.size(), carry_over_capacity_);
  std::vector<float> carry_over_audio;
  carry_over_audio.reserve(carry_over_samples);
  carry_over_audio.insert(carry_over_audio.begin(), data.end() - carry_over_samples, 
                            data.end());
  data.clear();
  data.insert(data.begin(), carry_over_audio.begin(), carry_over_audio.end());
}

void BatchedBuffer::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  audio_buffer_.clear();
}


} // end of namespace whisper
