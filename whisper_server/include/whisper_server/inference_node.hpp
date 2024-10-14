#ifndef WHISPER_NODES__INFERENCE_NODE_HPP_
#define WHISPER_NODES__INFERENCE_NODE_HPP_

#include <chrono>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <chrono>
//  reading writing files
#include <iostream>
#include <fstream>
#include <array>
#include <string_view>  // For compile-time string views
#include <mutex>


#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/int16_multi_array.hpp"
#include "std_msgs/msg/string.hpp"
#include "rcl_interfaces/msg/parameter_descriptor.hpp"

#include "whisper_idl/action/inference.hpp"
#include "whisper_idl/msg/audio_data.hpp" 
#include "whisper_idl/msg/whisper_chatter.hpp" 

#include "whisper_util/audio_buffers.hpp"
#include "whisper_util/model_manager.hpp"
#include "whisper_util/whisper.hpp"
#include "whisper_util/transcript_data.hpp"


#define WRITE_TEXT_PROBS_TO_FILE 1

namespace whisper {

using namespace std::chrono_literals;

class InferenceNode {
  using Inference = whisper_idl::action::Inference;
  using GoalHandleInference = rclcpp_action::ServerGoalHandle<Inference>;
public:
  InferenceNode(const rclcpp::Node::SharedPtr node_ptr);

protected:
  rclcpp::Node::SharedPtr node_ptr_;

  // parameters
  int step_ms_;
  int length_ms_;
  void declare_parameters_();
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr on_parameter_set_handle_;
  rcl_interfaces::msg::SetParametersResult
  on_parameter_set_(const std::vector<rclcpp::Parameter> &parameters);

  // audio subscription
  rclcpp::Subscription<whisper_idl::msg::AudioData>::SharedPtr audio_sub_;
  void on_audio_(const whisper_idl::msg::AudioData::SharedPtr msg);

  // action server
  // std::shared_ptr<GoalHandleInference> active_goal_;
  rclcpp_action::Server<Inference>::SharedPtr inference_action_server_;
  rclcpp_action::GoalResponse on_inference_(const rclcpp_action::GoalUUID &uuid,
                                            std::shared_ptr<const Inference::Goal> goal);
  rclcpp_action::CancelResponse
  on_cancel_inference_(const std::shared_ptr<GoalHandleInference> goal_handle);
  void on_inference_accepted_(const std::shared_ptr<GoalHandleInference> goal_handle);
  std::string inference_(const std::vector<float> &audio);
  rclcpp::Time inference_start_time_;

  // publisher
  bool active_;
  // std::vector<std::pair<rclcpp::Time, std::string>> transcript;
  rclcpp::Publisher<whisper_idl::msg::WhisperChatter>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  void timer_callback();

  // whisper audio data storage
  // Map of capture_ids to audio buffers.  -1 is default capture device
  std::mutex mutex_;
  size_t step_samples_;
  size_t length_samples;
  std::unordered_map<int, size_t> new_samples;   
  std::unordered_map<int, std::unique_ptr<BatchedBuffer>> 
                                    in_buffer_map_;     // Enqueue into this
  // Allow over-writing of audio_data_, so contains a sliding window
  std::unordered_map<int, std::unique_ptr<RingBuffer<float>>> 
                                    audio_data_map_;        // Dequeue into this


  // std::unique_ptr<BatchedBuffer> batched_buffer_;       
  // std::vector<float> new_audio_data_;                   // Dequeue into this
  // std::unique_ptr<RingBuffer<float>> audio_data_;       // Run inference on this full array

  // whisper
  std::unique_ptr<ModelManager> model_manager_;
  std::unique_ptr<Whisper> whisper_;
  std::string language_;
  void initialize_whisper_();

  // Transcription 
  // TranscriptUpdater updater;
  TranscriptData transcript;
  TranscriptionData best_transcript;
  int last_update_idx;
  int update_idx;
  std::pair<std::string, float> try_combine(const std::vector<std::string>& texts, 
                                        const std::vector<float>& probs, size_t& i);


  bool handle_inference();
  void initialize_data(const int &capture_id);
};

/* Helper function */
template<typename T> void declare_param(
  std::shared_ptr<rclcpp::Node> node_ptr, 
  const std::string& param_name, 
  const T& default_value, 
  const std::string& description = "") 
{
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.description = description;
  node_ptr->declare_parameter(param_name, default_value, descriptor);
}



} // end of namespace whisper
#endif // WHISPER_NODES__INFERENCE_NODE_HPP_
