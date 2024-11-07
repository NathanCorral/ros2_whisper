#include "whisper_util/transcript_data.hpp"

namespace whisper {

void Transcript::run(const Operations &operations, const std::vector<Word> &words_other) {
  // Increment when inserting, decrement when deleting from array
  int op_id_offset = 0; 
  for (const auto &op : operations) {
    switch (op.op_type_) {
      case OperationType::INCREMENT:
        inc_word(op.id_ + op_id_offset);
        break;
      case OperationType::DECREMENT:
        dec_word(op.id_ + op_id_offset);
        break;
      case OperationType::INSERT:
        insert_word(op.id_ + op_id_offset, words_other, op.other_id_);
        op_id_offset++;
        break;
      case OperationType::CONFLICT:
        conflict_merge_word(op.id_ + op_id_offset, words_other, op.other_id_);
        break;
      case OperationType::REMOVE:
        remove_word(op.id_ + op_id_offset);
        op_id_offset--;
        break;
      case OperationType::MATCHED_WORD:
        conflict_merge_word(op.id_ + op_id_offset, words_other, op.other_id_);
        inc_word(op.id_ + op_id_offset);
        break;
      case OperationType::MERGE_SEGMENTS:
        // printf("Merge segment\n");
        merge_word_segments(op.id_ + op_id_offset, words_other, op.other_id_);
        inc_word(op.id_ + op_id_offset);
        break;
      case OperationType::ADD_SEGMENT:
        // printf("Add segment\n");
        if ( add_word_segments(op.id_ + op_id_offset, words_other, op.other_id_) ) {
          op_id_offset++;
        }
        break;
      case OperationType::REMOVE_SEGMENT:
        // printf("Remove segment\n");
        if ( delete_word_segments(op.id_ + op_id_offset) ) {
          op_id_offset--;
        }
        break;
    }
  }
}

void Transcript::run(const Operations &operations) {
  int op_id_offset = 0; 
  for (const auto &op : operations) {
    switch (op.op_type_) {
      case OperationType::INCREMENT:
        inc_word(op.id_ + op_id_offset);
        break;
      case OperationType::DECREMENT:
        dec_word(op.id_ + op_id_offset);
        break;
      case OperationType::REMOVE:
        remove_word(op.id_ + op_id_offset);
        op_id_offset--;
        break;
      case OperationType::REMOVE_SEGMENT:
        if ( delete_word_segments(op.id_ + op_id_offset) ) {
          op_id_offset--;
        }
        break;
      default:
        break;
    }
  }
}

// Standard Operations
void Transcript::inc_word(const int id) {
  transcript_[id + stale_id_].inc_best();
}

void Transcript::dec_word(const int id) {
  transcript_[id + stale_id_].dec_best();
}

void Transcript::remove_word(const int id) {
  transcript_.erase(transcript_.begin() + id + stale_id_);
  // When removing a word, we need to adjust the pointers to segments
  for (size_t i = 0; i < segment_ids.size(); ++i) {
    if (segment_ids[i] > (id + stale_id_)) {
      --segment_ids[i];
    }
  }
}

void Transcript::insert_word(const int id,
                const std::vector<Word> &new_words,
                const int new_word_id) {
  transcript_.insert(transcript_.begin() + id + stale_id_, new_words[new_word_id]);
  // When inserting a word, we need to adjust the pointers to segments
  for (size_t i = 0; i < segment_ids.size(); ++i) {
    if (segment_ids[i] > (id + stale_id_)) {
      ++segment_ids[i];
    }
  }
}

void Transcript::conflict_merge_word(const int id,
                  const std::vector<Word> &no_conflict_other,
                  const int other_id) {
  transcript_[id + stale_id_].compare(no_conflict_other[other_id]);
}

void Transcript::merge_word_segments(const int id,
                  const std::vector<Word> &new_words,
                  const int other_id) {
  // Saftey check that the segment after does not have an earlier start
  auto seg_tscript_id_before = get_seg_before(id + stale_id_);
  auto seg_tscript_id_after = get_seg_after(id + stale_id_);
  if (seg_tscript_id_before >= 0 && 
        transcript_[seg_tscript_id_before].get_start() > new_words[other_id].get_start()) {
    fprintf(stderr, "Unable to Merge Segment. Earlier Segment has later timestamp\n");
    return;
  }
  if (seg_tscript_id_after >= 0 && 
        transcript_[seg_tscript_id_after].get_start() < new_words[other_id].get_start()) {
    fprintf(stderr, "Unable to Merge Segment. Next Segment has earlier timestamp\n");
    return;
  }

  // printf("MERGE Segment %s    ::with::  %s\n", transcript_[id + stale_id_].as_str().c_str(),
  //                                             new_words[other_id].as_str().c_str());
  transcript_[id + stale_id_].merge_segments(new_words[other_id]);

  // Fix the duration of the segment before (timestamp may have changed):
  if (seg_tscript_id_before >= 0) {
    auto new_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
              new_words[other_id].get_start() - transcript_[seg_tscript_id_before].get_start());
    // printf("Change Previous segment druation from:  %ld ->  %ld\n", 
    //             transcript_[seg_tscript_id_before].get_duration().count(), new_duration.count());
    transcript_[seg_tscript_id_before].set_duration(new_duration);
  }

  // Adjust the duration of the current segment 
  if (seg_tscript_id_after >= 0) {
    auto new_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
              transcript_[seg_tscript_id_after].get_start() - transcript_[id + stale_id_].get_start());
    // printf("Adjust current druation from:  %ld ->  %ld\n", 
    //             transcript_[id + stale_id_].get_duration().count(), new_duration.count());
    transcript_[id + stale_id_].set_duration(new_duration);
  }
}

bool Transcript::add_word_segments(const int id,
                  const std::vector<Word> &new_words,
                  const int other_id) {
  // valid check.  id + stale_id_ == transcript_.size() when pushing back
  if ( ! (id + stale_id_ >= 0 && static_cast<size_t>(id + stale_id_) <= transcript_.size()) ) {
    fprintf(stderr, "Add segment transcript bound check failed, "
                  "id (abs):  %d, stale_id:  %d, Transciript size:  %ld\n",
                  (id + stale_id_), stale_id_, transcript_.size());
    return false;
  }
  else if ( other_id < 0 || static_cast<size_t>(other_id) >= new_words.size() || 
              ! (new_words[other_id].is_segment()) ) {
    fprintf(stderr, "Add segment other id bounds check failed\n");
    return false;
  }

  // Get transcript_ id's of the segments before and after
  auto seg_tscript_id_before = get_seg_before(id + stale_id_);
  auto seg_tscript_id_after = get_seg_after(id + stale_id_);

  // Check if there is a segment nearby
  if (seg_tscript_id_after >= 0 && id + stale_id_ + 2 >= seg_tscript_id_after) {
    printf("Found close segment (after), skipping...\n");
    // merge_word_segments(seg_tscript_id_after - stale_id_, new_words, other_id);
    return false;
  }
  if (seg_tscript_id_before >= 0 && id + stale_id_  <= seg_tscript_id_before + 2) {
    printf("Found close segment (before), skipping...\n");
    // merge_word_segments(seg_tscript_id_before - stale_id_, new_words, other_id);
    return false;
  }

  // Saftey check that the timestamps are valid
  if (seg_tscript_id_before > 0 && 
        transcript_[seg_tscript_id_before].get_start() > new_words[other_id].get_start()) {
    fprintf(stderr, "Unable to insert segment with earlier timestamp than previous segement\n");
    return false;
  }
  if (seg_tscript_id_after > 0 && 
        transcript_[seg_tscript_id_after].get_start() < new_words[other_id].get_start()) {
    fprintf(stderr, "Unable to insert segment with later timestamp than next segement\n");
    return false;
  }

  // Debug Print info on the segments:
  {
    // printf("ADDING SEGMENT:   %s\n", new_words[other_id].as_str().c_str());
    // if (seg_tscript_id_before >= 0) {
    //   printf("\tSegment Before:   %s\n", transcript_[seg_tscript_id_before].as_str().c_str());
    // }
    // if (seg_tscript_id_after >= 0) {
    //   printf("\tSegment After:   %s\n", transcript_[seg_tscript_id_after].as_str().c_str());
    // }
  }
  // Perform insert or push_back
  insert_word(id, new_words, other_id);

  if (seg_tscript_id_before >= 0) {
    auto new_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
              new_words[other_id].get_start() - transcript_[seg_tscript_id_before].get_start());
    // printf("Change Previous segment druation from:  %ld ->  %ld\n", 
    //       transcript_[seg_tscript_id_before].get_duration().count(), new_duration.count());
    transcript_[seg_tscript_id_before].set_duration(new_duration);
  }

  if ( !(seg_tscript_id_after >= 0) ) {
    // If there is no segment after, set the previous seg duration based on timestamp
    // Insert new segment id
    segment_ids.push_back(id + stale_id_);
    return true;
  } 
  // Adjust the duration of the current segment
  auto new_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            transcript_[seg_tscript_id_after].get_start() - transcript_[id + stale_id_].get_start());
  // printf("Adjust current druation from:  %ld ->  %ld\n", 
  //             transcript_[id + stale_id_].get_duration().count(), new_duration.count());
  transcript_[id + stale_id_].set_duration(new_duration);
  
  // Insert the index of the added segment into sorted segment_ids array
  auto it = std::lower_bound(segment_ids.begin(), segment_ids.end(), id + stale_id_);
  {
    // if (it != segment_ids.end()) {
    //   printf("\t\t- inserting word segment: %d: '%s...'  at seg idx: %ld after:  %d\n", 
    //                                                             id + stale_id_, 
    //       static_cast<size_t>(id + stale_id_ + 1) < transcript_.size() ? 
    //                         transcript_[id + stale_id_ + 1].get().c_str() : ":::END",
    //                                                             std::distance(segment_ids.begin(), it), 
    //                                                             *it);
    // }  else {
    //   printf("\t\t- inserting word id %d:  '%s...' at end of segment_ids\n", 
    //                                                           id + stale_id_,
    //       static_cast<size_t>(id + stale_id_ + 1) < transcript_.size() ? 
    //                         transcript_[id + stale_id_ + 1].get().c_str() : ":::END");
    // } 
  }
  segment_ids.insert(it, id + stale_id_);
  return true;
}

bool Transcript::delete_word_segments(const int id) {
  // valid check.  id + stale_id_ == transcript_.size() when pushing back
  if ( ! (id + stale_id_ >= 0 && static_cast<size_t>(id + stale_id_) < transcript_.size()) ) {
    fprintf(stderr, "Bounds check fail on deleting segment\n");
    return false;
  }
  else if ( ! (transcript_[id + stale_id_].is_segment()) ) {
    fprintf(stderr, "Attempt to delete non-segment\n");
    return false;
  }
  auto it = std::lower_bound(segment_ids.begin(), segment_ids.end(), id + stale_id_);
  if (it == segment_ids.end()) {
    fprintf(stderr, "Segment id value not found in segment_ids\n");
    return false;
  }

  // Adjust duration of previous segment
  auto seg_tscript_id_before = get_seg_before(id + stale_id_);
  if (seg_tscript_id_before >= 0) {
    // printf("Change Previous segment druation from:  %ld ->  %ld\n", 
    //         transcript_[seg_tscript_id_before].get_duration().count(), 
    //         (transcript_[seg_tscript_id_before].get_duration() + 
    //                 transcript_[id + stale_id_].get_duration()).count());
    transcript_[seg_tscript_id_before].set_duration(
        transcript_[seg_tscript_id_before].get_duration() + 
        transcript_[id + stale_id_].get_duration());
  }

  // Remove segment from transcript
  remove_word(id);

  // Remove index of segment from (sorted) segment_ids array
  segment_ids.erase(it);
  return true;
}


// Other Functions
void Transcript::push_back(const std::vector<Word> &words_and_segments) {
  Transcript::Operations pending_ops;
  for (size_t i = 0; i < words_and_segments.size(); ++i) {
    if (words_and_segments[i].is_segment()) {
      pending_ops.push_back({ADD_SEGMENT, static_cast<int>(transcript_.size()) - stale_id_, 
                                                              static_cast<int>(i)});
    } else {
      pending_ops.push_back({INSERT, static_cast<int>(transcript_.size()) - stale_id_, 
                                                              static_cast<int>(i)});
    }
  }
  run(pending_ops, words_and_segments);
}

void Transcript::clear_mistakes(const int occurrence_threshold) {
  Transcript::Operations pending_ops;
  for (int id = stale_id_; id < static_cast<int>(transcript_.size()); ++id) {
    if (transcript_[id].get_occurrences() <= occurrence_threshold) {
      if (transcript_[id].is_segment()) {
        pending_ops.push_back({REMOVE_SEGMENT, id-stale_id_});
      } else {
        pending_ops.push_back({REMOVE, id-stale_id_});
      }
    }
  }
  run(pending_ops);
}


int Transcript::get_seg_before(const int id) {
  int ret = -1;
  // printf("segids size:  %ld,  Looking for %d in [", segment_ids.size(), id);
  for (const auto seg_id : segment_ids) {
    // printf("%d, ", seg_id);

    if (id <= seg_id) {
      break;
    }
    ret = seg_id;
  }
  // printf("] -- Found:  %d\n", ret);
  return ret;
}

int Transcript::get_seg_after(const int id) {
  int ret = -1;
  for (size_t i = 1; i < segment_ids.size(); ++i) {
    if (id < segment_ids[i]) {
      ret = segment_ids[i];
      break;
    }
  }
  return ret;
}

} // end of namespace whisper
