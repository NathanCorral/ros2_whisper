#include "whisper_nodes/listen_node.hpp"

namespace whisper {
ListenNode::ListenNode(const rclcpp::Node::SharedPtr node_ptr)
    : node_ptr_(node_ptr), listening_(false), audio_buffer_(100) {

  audio_subscription_ = node_ptr_->create_subscription<std_msgs::msg::Int16MultiArray>(
      "audio", 10, [this](const std_msgs::msg::Int16MultiArray::SharedPtr msg) {
        if (listening_) {
          audio_buffer_.ring_buffer.enqueue(msg->data);
        }
      });

  listen_action_server_ = rclcpp_action::create_server<ListenAction>(
      node_ptr_, "listen",
      std::bind(&ListenNode::on_listen_goal_, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&ListenNode::on_listen_cancel_, this, std::placeholders::_1),
      std::bind(&ListenNode::on_listen_goal_accepted_, this, std::placeholders::_1));
}

rclcpp_action::GoalResponse
ListenNode::on_listen_goal_(const rclcpp_action::GoalUUID &uuid,
                            const ListenAction::Goal::ConstSharedPtr goal) {
  if (listening_) {
    return rclcpp_action::GoalResponse::REJECT;
  }
  // clear?
  listening_ = true;
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse
ListenNode::on_listen_cancel_(const std::shared_ptr<ListenGoalHandle> goal_handle) {
  listening_ = false;
  return rclcpp_action::CancelResponse::ACCEPT;
}

void ListenNode::on_listen_goal_accepted_(const std::shared_ptr<ListenGoalHandle> goal_handle) {
  std::thread listen_thread(&ListenNode::listen_, this, goal_handle);
  listen_thread.detach();
}

void ListenNode::listen_(const std::shared_ptr<ListenGoalHandle> goal_handle) {
  RCLCPP_INFO(node_ptr_->get_logger(), "Listening...");
  auto feedback = std::make_shared<ListenAction::Feedback>();
  auto result = std::make_shared<ListenAction::Result>();
  start_time_ = node_ptr_->now();
  auto max_duration = goal_handle->get_goal()->max_duration;

  std_msgs::msg::MultiArrayDimension dim;
  dim.label = "audio";
  dim.stride = 1;
  feedback->audio_buffer.layout.data_offset = 0;
  feedback->audio_buffer.layout.dim.push_back(dim);

  while (listening_) {
    if (audio_buffer_.ring_buffer.is_full()) {
      RCLCPP_WARN(node_ptr_->get_logger(), "Audio buffer is full!");
    }

    if (!audio_buffer_.ring_buffer.has_data()) {
      continue;
    }
    feedback->audio_buffer.data = AudioBuffer::normalize(audio_buffer_.ring_buffer.dequeue());
    feedback->audio_buffer.layout.dim[0].size = feedback->audio_buffer.data.size();
    goal_handle->publish_feedback(feedback);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (node_ptr_->now() - start_time_ > max_duration) {
      listening_ = false;
      break;
    }
  }

  listening_ = false;
}
} // end of namespace whisper
