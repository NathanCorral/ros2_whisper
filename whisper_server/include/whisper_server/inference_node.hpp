#ifndef WHISPER_NODES__INFERENCE_NODE_HPP_
#define WHISPER_NODES__INFERENCE_NODE_HPP_

#include <chrono>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/int16_multi_array.hpp"
#include "std_msgs/msg/string.hpp"
#include "rcl_interfaces/msg/parameter_descriptor.hpp"

#include "whisper_idl/action/inference.hpp"
#include "whisper_util/audio_buffers.hpp"
#include "whisper_util/model_manager.hpp"
#include "whisper_util/whisper.hpp"

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
  void declare_parameters_();
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr on_parameter_set_handle_;
  rcl_interfaces::msg::SetParametersResult
  on_parameter_set_(const std::vector<rclcpp::Parameter> &parameters);

  // audio subscription
  rclcpp::Subscription<std_msgs::msg::Int16MultiArray>::SharedPtr audio_sub_;
  void on_audio_(const std_msgs::msg::Int16MultiArray::SharedPtr msg);

  // action server
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
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  void timer_callback();

  // whisper audio data storage
  int length_ms_;
  size_t step_samples_;
  std::unique_ptr<BatchedBuffer> batched_buffer_; // Enqueue into this
  std::vector<float> new_audio_data_; // Dequeue into this
  std::unique_ptr<RingBuffer<float>> audio_data_; // Run inference on this full array

  // whisper
  std::unique_ptr<ModelManager> model_manager_;
  std::unique_ptr<Whisper> whisper_;
  std::string language_;
  void initialize_whisper_();
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
