#include "whisper_server/inference_node.hpp"

namespace whisper {
InferenceNode::InferenceNode(const rclcpp::Node::SharedPtr node_ptr)
    : node_ptr_(node_ptr), language_("en") {
  declare_parameters_();

  auto cb_group = node_ptr_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions options;
  options.callback_group = cb_group;

  // audio subscription
  audio_sub_ = node_ptr_->create_subscription<std_msgs::msg::Int16MultiArray>(
      "audio", 5, std::bind(&InferenceNode::on_audio_, this, std::placeholders::_1), options);

  // inference action server
  inference_action_server_ = rclcpp_action::create_server<Inference>(
      node_ptr_, "inference",
      std::bind(&InferenceNode::on_inference_, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&InferenceNode::on_cancel_inference_, this, std::placeholders::_1),
      std::bind(&InferenceNode::on_inference_accepted_, this, std::placeholders::_1));

  // parameter callback handle
  on_parameter_set_handle_ = node_ptr_->add_on_set_parameters_callback(
      std::bind(&InferenceNode::on_parameter_set_, this, std::placeholders::_1));

  // whisper data
  batched_buffer_ = std::make_unique<BatchedBuffer>(
      std::chrono::seconds(node_ptr_->get_parameter("buffer_capacity").as_int()),
      std::chrono::milliseconds(node_ptr_->get_parameter("carry_over_capacity").as_int()));
  step_ms_ = node_ptr_->get_parameter("step_ms").as_int();
  step_samples_ = step_ms_ * WHISPER_SAMPLE_RATE / 1e3;
  length_ms_ = node_ptr_->get_parameter("length_ms").as_int();
  const std::size_t length_samples = length_ms_ * WHISPER_SAMPLE_RATE / 1e3;
  audio_data_ = std::make_unique<RingBuffer<float>>(length_samples);
  audio_data_->enqueue(std::vector<float>(length_samples, 0.0f));

  // whisper
  model_manager_ = std::make_unique<ModelManager>();
  whisper_ = std::make_unique<Whisper>();

  initialize_whisper_();

  // publisher
  publisher_ = node_ptr_->create_publisher<std_msgs::msg::String>("audio_transcript", 10);
  timer_ = node_ptr_->create_wall_timer(
      50ms, std::bind(&InferenceNode::timer_callback, this));
  active_ = node_ptr_->get_parameter("active").as_bool();
}

void InferenceNode::declare_parameters_() {
  // Data/Buffer parameters
  declare_param(node_ptr_, "buffer_capacity", 2, 
                        "Capacity of the incomming audio buffer in seconds.");
  declare_param(node_ptr_, "carry_over_capacity", 200, 
                        "audio to keep from previous step in ms.");
  declare_param(node_ptr_, "length_ms", 3000, 
                        "Length of (previous) audio data to process together as a batch.");
  declare_param(node_ptr_, "step_ms", 3000, 
                        "Publish/give feedback every step_ms in ms.");

  // whisper parameters
  declare_param(node_ptr_, "model_name", "base.en", 
              "Name of the Whisper model to be used.");
  declare_param(node_ptr_, "wparams.language", "en", 
              "Language code for Whisper's language model.");
  declare_param(node_ptr_, "wparams.n_threads", 4, 
              "Number of threads used by Whisper.");
  declare_param(node_ptr_, "wparams.print_progress", false, 
              "Whether to print progress updates during Whisper processing.");
  declare_param(node_ptr_, "cparams.flash_attn", true, 
              "Use flash attention for optimized GPU inference.");
  declare_param(node_ptr_, "cparams.gpu_device", 0, 
              "The GPU device to be used for processing.");
  declare_param(node_ptr_, "cparams.use_gpu", true, 
              "Enable or disable GPU acceleration.");

  // Control actively publishing the transcript
  declare_param(node_ptr_, "active", false, 
    "Control whether the node is actively publishing the live inference.");
}

void InferenceNode::initialize_whisper_() {
  std::string model_name = node_ptr_->get_parameter("model_name").as_string();
  RCLCPP_INFO(node_ptr_->get_logger(), "Checking whether model %s is available...",
              model_name.c_str());
  if (!model_manager_->is_available(model_name)) {
    RCLCPP_INFO(node_ptr_->get_logger(), "Model %s is not available. Attempting download...",
                model_name.c_str());
    if (model_manager_->make_available(model_name) != 0) {
      std::string err_msg = "Failed to download model " + model_name + ".";
      RCLCPP_ERROR(node_ptr_->get_logger(), err_msg.c_str());
      throw std::runtime_error(err_msg);
    }
    RCLCPP_INFO(node_ptr_->get_logger(), "Model %s downloaded.", model_name.c_str());
  }
  RCLCPP_INFO(node_ptr_->get_logger(), "Model %s is available.", model_name.c_str());

  language_ = node_ptr_->get_parameter("wparams.language").as_string();
  whisper_->wparams.language = language_.c_str();
  whisper_->wparams.n_threads = node_ptr_->get_parameter("wparams.n_threads").as_int();
  whisper_->wparams.print_progress = node_ptr_->get_parameter("wparams.print_progress").as_bool();
  whisper_->cparams.flash_attn = node_ptr_->get_parameter("cparams.flash_attn").as_bool();
  whisper_->cparams.gpu_device = node_ptr_->get_parameter("cparams.gpu_device").as_int();
  whisper_->cparams.use_gpu = node_ptr_->get_parameter("cparams.use_gpu").as_bool();

  RCLCPP_INFO(node_ptr_->get_logger(), "Initializing model %s...", model_name.c_str());
  whisper_->initialize(model_manager_->get_model_path(model_name));
  RCLCPP_INFO(node_ptr_->get_logger(), "Model %s initialized.", model_name.c_str());
}

rcl_interfaces::msg::SetParametersResult
InferenceNode::on_parameter_set_(const std::vector<rclcpp::Parameter> &parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  for (const auto &parameter : parameters) {
    if (parameter.get_name() == "n_threads") {
      whisper_->wparams.n_threads = parameter.as_int();
      RCLCPP_INFO(node_ptr_->get_logger(), "Parameter %s set to %d.", parameter.get_name().c_str(),
                  whisper_->wparams.n_threads);
      continue;
    }
    if (parameter.get_name() == "active") {
      // Abort goal if becoming active and current action server
      // if (parameter.as_bool() && active_goal_) {
      //   RCLCPP_WARN(node_ptr_->get_logger(), "Aborting current goal.  Subscribe to %s", 
      //                             publisher_->get_topic_name());
      //   active_goal_->abort(std::make_shared<Inference::Result>());
      // }
      // Set new parameter
      active_ = parameter.as_bool();
      RCLCPP_INFO(node_ptr_->get_logger(), "Parameter %s set to %d.", parameter.get_name().c_str(),
                  active_);
      continue;
    }
    result.reason = "Parameter " + parameter.get_name() + " not handled.";
    result.successful = false;
    RCLCPP_WARN(node_ptr_->get_logger(), result.reason.c_str());
  }
  result.successful = true;
  return result;
}

void InferenceNode::on_audio_(const std_msgs::msg::Int16MultiArray::SharedPtr msg) {
  batched_buffer_->enqueue(msg->data);
}

void InferenceNode::timer_callback()
{
  if (!active_) {
    return;
  }

  // While dequeue-ing, data is automatically removed from the internal buffer
  batched_buffer_->dequeue(new_audio_data_);
  if (new_audio_data_.size() <  step_samples_) {
    return;
  }

  // Ring buffer, never dequeue so it stays full for inference
  audio_data_->enqueue(new_audio_data_);
  // after adding, clear and role over audio to help with word breaks. 
  batched_buffer_->clear_and_carry_over_(new_audio_data_);

  auto transcription = inference_(audio_data_->peak());

  auto message = std_msgs::msg::String();
  message.data = transcription;
  publisher_->publish(message);
}

rclcpp_action::GoalResponse
InferenceNode::on_inference_(const rclcpp_action::GoalUUID & /*uuid*/,
                             std::shared_ptr<const Inference::Goal> /*goal*/) {
  if (active_) {
    RCLCPP_WARN(node_ptr_->get_logger(), "Currently publishing from input:   %s.  Subscribe to %s", 
      audio_sub_->get_topic_name(), publisher_->get_topic_name());
    return rclcpp_action::GoalResponse::REJECT;
  }

  // if (active_goal_) {
  //   RCLCPP_INFO(node_ptr_->get_logger(), "Preempting the currently active goal.");
  //   active_goal_->abort(std::make_shared<Inference::Result>());
  // }

  RCLCPP_INFO(node_ptr_->get_logger(), "Received inference request.");
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse
InferenceNode::on_cancel_inference_(const std::shared_ptr<GoalHandleInference> /*goal_handle*/) {
  RCLCPP_INFO(node_ptr_->get_logger(), "Cancelling inference...");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void InferenceNode::on_inference_accepted_(const std::shared_ptr<GoalHandleInference> goal_handle) {
  RCLCPP_INFO(node_ptr_->get_logger(), "Starting inference...");
  auto feedback = std::make_shared<Inference::Feedback>();
  auto result = std::make_shared<Inference::Result>();
  inference_start_time_ = node_ptr_->now();
  // active_goal_ = goal_handle;
  int batch_idx = 0;

  while (rclcpp::ok()) {
    if (node_ptr_->now() - inference_start_time_ > goal_handle->get_goal()->max_duration) {
      result->info = "Inference timed out.";
      RCLCPP_INFO(node_ptr_->get_logger(), result->info.c_str());
      goal_handle->succeed(result);
      batched_buffer_->clear();
      return;
    }

    if (goal_handle->is_canceling()) {
      result->info = "Inference cancelled.";
      RCLCPP_INFO(node_ptr_->get_logger(), result->info.c_str());
      goal_handle->canceled(result);
      batched_buffer_->clear();
      return;
    }

    if (active_) {
      result->info = "Action server stopped.  Subscribe to continuously published topic.";
      RCLCPP_INFO(node_ptr_->get_logger(), result->info.c_str());
      goal_handle->succeed(result);
      batched_buffer_->clear();
      return;
    }

    // While dequeue-ing, data is automatically removed from the internal buffer
    batched_buffer_->dequeue(new_audio_data_);
    if (new_audio_data_.size() < step_samples_) {
      rclcpp::sleep_for(1ms);
      continue;
    }

    // Ring buffer, never dequeue so it stays full
    audio_data_->enqueue(new_audio_data_);
    // after adding, clear and role over audio to help with word breaks. 
    batched_buffer_->clear_and_carry_over_(new_audio_data_);

    auto transcription = inference_(audio_data_->peak());

    // feedback to client
    feedback->transcription = transcription;
    feedback->batch_idx = batch_idx++;
    goal_handle->publish_feedback(feedback);

    // update inference result
    result->transcriptions.push_back(feedback->transcription);
  }

  if (rclcpp::ok()) {
    result->info = "Inference succeeded.";
    RCLCPP_INFO(node_ptr_->get_logger(), result->info.c_str());
    goal_handle->succeed(result);
    batched_buffer_->clear();
  }

  // active_goal_.reset();
}

std::string InferenceNode::inference_(const std::vector<float> &audio) {
  auto inference_start_time = node_ptr_->now();
  auto transcription = whisper_->forward(audio);
  auto inference_duration =
      (node_ptr_->now() - inference_start_time).to_chrono<std::chrono::milliseconds>();
  if (inference_duration > whisper::count_to_time(audio.size())) {
    RCLCPP_WARN(node_ptr_->get_logger(),
                "Inference took longer than audio buffer size. This leads to un-inferenced audio "
                "data. Consider increasing thread number or compile with accelerator support.");
  }
  //   std::vector<std::pair<rclcpp::Time, std::string>> transcript;
  // transcript.push_back(std::make_pair(inference_start_time, transcription));

// std::vector<whisper_token>
  std::vector<std::string> texts;
  std::vector<float> probs;
  RCLCPP_WARN(node_ptr_->get_logger(), "Getting PRobabilities\n");
  auto p = whisper_->p(texts, probs);
  // Print out the size of texts and probs arrays:
  RCLCPP_INFO(node_ptr_->get_logger(), "texts size %ld,   probs size %ld", texts.size(), probs.size());

  // loop through the text and probabilities arrays,  they should be the same size but check in case
  std::ofstream myfile;
  std::ofstream myprobs;
  // append to file "test.txt"
  myfile.open("test_text.txt", std::ios_base::app); 
  myprobs.open("test_probs.csv", std::ios_base::app); 

  // Use a stringstream to accumulate and print out the texts
  // std::stringstream ss;
  myfile << "{";
  for (size_t i = 0; i < texts.size(); i++) {
    // If the text starts with "[_" and ends with "]", 
    //  then it is a whisper specific token so skip it
    if (texts[i].substr(0, 2) == "[_" && texts[i].substr(texts[i].size() - 1, 1) == "]") {
      continue;
    }
    myfile << '"' << texts[i] << '"';
    myprobs << probs[i];
    if (i < texts.size() - 2) {
      myfile << ", ";
      myprobs << ",";
    }
  }
  myfile << "}\n";
  myprobs << "\n";
  // RCLCPP_INFO(node_ptr_->get_logger(), "%s\n", ss.str().c_str());
  // for (size_t i = 0; i < texts.size(); i++) {
  //   std::string t = texts[i];
  //   const float  p = probs[i];
  //   // print out the texts and the probability
  //   RCLCPP_INFO(node_ptr_->get_logger(), "%f>>> %s\n", p, t.c_str());
  // }
  // printf("%f>>> %s\n", p, text);

  return transcription;
}


} // end of namespace whisper
