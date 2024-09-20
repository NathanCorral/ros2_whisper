#ifndef WHISPER_UTIL__AUDIO_BUFFERS_HPP_
#define WHISPER_UTIL__AUDIO_BUFFERS_HPP_

#include <chrono>
#include <limits>
#include <mutex>
#include <vector>

// #include <ostream>
#include <iostream>

#include "whisper.h"

namespace whisper {
inline std::size_t time_to_count(const std::chrono::milliseconds &ms) {
  return ms.count() * WHISPER_SAMPLE_RATE / 1e3;
};

inline std::chrono::milliseconds count_to_time(const std::size_t &count) {
  return std::chrono::milliseconds(count * static_cast<std::size_t>(1e3) / WHISPER_SAMPLE_RATE);
};

/**
 * @brief A ring buffer implementation. This buffer is **not** thread-safe. It is the user's
 * responsibility to ensure thread-safety.
 *
 * @tparam value_type
 */
template <typename value_type> class RingBuffer {
  using const_reference = const value_type &;

public:
  RingBuffer(const std::size_t &capacity);

  void enqueue(const_reference data);
  void enqueue(const std::vector<value_type> data);

  // Returns all data in the buffer
  std::vector<value_type> peak();

  value_type dequeue();
  inline bool is_full() const { return size_ == capacity_; }
  void clear();

  inline const std::size_t &capacity() const { return capacity_; }
  inline const std::size_t &size() const { return size_; }

protected:
  void increment_head_();
  void increment_tail_();

  std::size_t capacity_;
  std::vector<value_type> buffer_;
  std::size_t head_;
  std::size_t tail_;
  std::size_t size_;
};

/**
 * @brief A thread-safe buffer for storing audio data. The user enqueues data from an audio stream
 * in thread A and dequeues data in thread B. 
 * 
 * Thread A: enqueue into audio_buffer_ (ring buffer)
 * Thread B: dequeue from audio_buffer_ and store into external array
 *
 * The buffer should be dequeued quicker than buffer_capacity_ to avoid loss of data.
 *
 */
class BatchedBuffer {
public:
  BatchedBuffer(
      const std::chrono::milliseconds &buffer_capacity = std::chrono::seconds(2),
      const std::chrono::milliseconds &carry_over_capacity = std::chrono::milliseconds(200));

  void enqueue(const std::vector<std::int16_t> &audio);

  // Dequeue data from the buffer and store it in the result vector
  void dequeue(std::vector<float> & result);

  // Clear the external data buffer, copy over the last carry_over_capacity ms of data
  void clear_and_carry_over_(std::vector<float> & data);

  void clear();
  inline const std::size_t &buffer_size() const { return audio_buffer_.size(); };

protected:

  std::mutex mutex_;

  std::size_t carry_over_capacity_;

  RingBuffer<std::int16_t> audio_buffer_;
};



/*
Templated function implementation.
*/
template <typename value_type>
RingBuffer<value_type>::RingBuffer(const std::size_t &capacity)
    : capacity_(capacity), buffer_(capacity) {
  clear();
};

template <typename value_type> void RingBuffer<value_type>::enqueue(const_reference data) {
  increment_head_();
  if (is_full()) {
    increment_tail_();
  }
  buffer_[head_] = data;
}

template <typename value_type> void RingBuffer<value_type>::enqueue(const std::vector<value_type> data) {
  for(const auto &d : data) {
    enqueue(d);
  }
}

template <typename value_type> value_type RingBuffer<value_type>::dequeue() {
  increment_tail_();
  return buffer_[tail_];
}

template <typename value_type> std::vector<value_type> RingBuffer<value_type>::peak() {
  std::vector<value_type> result; 
  result.reserve(size_);
  for (std::size_t i = 0; i < size_; ++i) {
    result.push_back(buffer_[(tail_ + i) % capacity_]);
  }
  return result;
}

template <typename value_type> void RingBuffer<value_type>::clear() {
  head_ = 0;
  tail_ = 0;
  size_ = 0;
}

template <typename value_type> void RingBuffer<value_type>::increment_head_() {
  ++head_;
  ++size_;
  if (head_ >= capacity_) {
    head_ = 0;
  }
}
template <typename value_type> void RingBuffer<value_type>::increment_tail_() {
  ++tail_;
  --size_;
  if (tail_ >= capacity_) {
    tail_ = 0;
  }
}

} // end of namespace whisper



#endif // WHISPER_UTIL__AUDIO_BUFFERS_HPP_
