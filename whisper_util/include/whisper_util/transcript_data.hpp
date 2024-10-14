#ifndef WHISPER_UTIL__TRANSCRIPT_DATA_HPP_
#define WHISPER_UTIL__TRANSCRIPT_DATA_HPP_

#include <vector>
#include <string>
#include <numeric> // accumulate
#include <stdexcept>
#include <cstdint>
#include <algorithm> // reverse
#include <tuple>

namespace whisper {



class WhisperSegment {
public:
  std::string text;
  int64_t t0;
  int64_t t1;
  int64_t audio_offset_ms;
  int occurances;
  
  // bool ends_with_speaker_turn;
  std::vector<int> counts;
  std::vector<float> probs;
  std::vector<std::string> tokens;
  
  // Map a starting index in text to its token index.
  std::vector<int> text_token_mapping;

  // When running clean, compare values with conflicts and take one with highest count
  std::vector<std::vector<int>> text_clonflict_mapping; // one-to-many mapping
  std::vector<std::string> token_confilicts;     // tokens over-written
  std::vector<int> token_confilict_counts;

  WhisperSegment(const std::string & text, int64_t t0, int64_t t1)
        : text(text), t0(t0), t1(t1), audio_offset_ms(0), occurances(1) {};

  WhisperSegment(const std::string & text, int64_t t0, int64_t t1, 
                      std::vector<std::string> tokens, std::vector<float> probs)
        : text(text), t0(t0), t1(t1), audio_offset_ms(0), occurances(1),
          tokens(tokens), probs(probs)
         {
            counts = std::vector<int>(tokens.size(), 1);  
            text_token_mapping = mapCharToToken(text, tokens);
         };

  std::tuple<double, int, int, int>  text_overlap(const WhisperSegment & other) const {
    return percentageOverlap(text, other.text);
  } 

  std::tuple<std::string, int, int> longestCommonSubstring(const std::string& str1, const std::string& str2) const {
      // return (lcs, start_str1, start_str2)
      int len1 = str1.size();
      int len2 = str2.size();
      
      // Create a 2D table to store lengths of longest common suffixes of substrings
      std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1, 0));
      
      // Length of the longest common substring
      int maxLength = 0;
      // Ending index of the longest common substring in str1
      int end_str1 = 0;
      // Ending index of the longest common substring in str2
      int end_str2 = 0;

      // Build the dp table
      for (int i = 1; i <= len1; ++i) {
          for (int j = 1; j <= len2; ++j) {
              // if (str1[i - 1] == str2[j - 1]) {
              if (std::tolower(static_cast<unsigned char>(str1[i - 1])) == 
                    std::tolower(static_cast<unsigned char>(str2[j - 1]))) {  
                  dp[i][j] = dp[i - 1][j - 1] + 1;
                  if (dp[i][j] > maxLength) {
                      maxLength = dp[i][j];
                      end_str1 = i - 1;
                      end_str2 = j - 1;
                  }
              }
          }
      }
      
      // If no common substring, return empty string
      if (maxLength == 0) {
        return std::make_tuple("", -1, -1);
      }

      // Calculate start index of the LCS in str1
      int start_str1 = end_str1 - maxLength + 1;
      int start_str2 = end_str2 - maxLength + 1;
      
      // Return the longest common substring
      return std::make_tuple(str1.substr(end_str1 - maxLength + 1, maxLength), start_str1, start_str2);
  };

  // Function to calculate percentage overlap
  // Return tuple of:
  //    perc / 100 overlap, 
  //    start position on str1 (calling), 
  //    start position on str2 (new), 
  //    length of lcs
  std::tuple<double, int, int, int> percentageOverlap(const std::string& str1, const std::string& str2) const {
      // return the lcsLength
      int start_str1;
      int start_str2;
      std::string lcs; 
      
      std::tie(lcs, start_str1, start_str2) = longestCommonSubstring(str1, str2);

      int lcsLength = lcs.size();
      
      if (lcsLength == 0) {
          return std::make_tuple(0.0, 0, 0, 0);
      }
      
      // Calculate overlap percentage based on the shorter string
      double overlapPercentage = (static_cast<double>(lcsLength) / std::min(str1.size(), str2.size()));
      
      return std::make_tuple(overlapPercentage, start_str1, start_str2, lcsLength);
  };

  // Function to map text indices to token indices
std::vector<int> mapCharToToken(const std::string& text, const std::vector<std::string>& tokens) {
    std::vector<int> charToTokenMap(text.size(), -1); // Initialize mapping to -1 (default)
    
    size_t textPos = 0;  // Current position in text
    for (size_t tokenIdx = 0; tokenIdx < tokens.size(); ++tokenIdx) {
      const std::string& token = tokens[tokenIdx];
      size_t foundPos = text.find(token, textPos);
      
      // If the token is found in text
      if (foundPos != std::string::npos) {
        // Mark all the characters of the token to map to the tokenIdx
        for (size_t i = 0; i < token.size(); ++i) {
            charToTokenMap[foundPos + i] = tokenIdx;
        }
        // Move textPos forward to avoid rechecking overlapping tokens
        textPos = foundPos + token.size();
      }
    }
    
    return charToTokenMap;
  };
};



class TranscriptionData {
public:


  void merge(WhisperSegment & cur, WhisperSegment & update, 
                int cur_start, int update_start, int lcs_len) {

    if (lcs_len == 0) {
      return;
    }

    // Skip to end of cur/update lcs.
    //  TODO:  Update token counts
    int cur_idx = cur_start + lcs_len;
    int update_idx = update_start + lcs_len;
    printf(" last same char / diff char: '%c / %c'\n", cur.text[cur_idx], update.text[update_idx]);

    
    int prev_token_idx = -1;

    // Increase the token reference and the count for each token in the lcs
    for (int i=0; i<lcs_len; i++) {
      auto cur_token_idx = cur.text_token_mapping[cur_start + i];
      if (prev_token_idx != cur_token_idx) {
        cur.counts[cur_token_idx]++;
        prev_token_idx = cur_token_idx;
      }
    }

    while (cur_idx < cur.text.size() && update_idx < update.text.size()) {
      auto cur_token_idx = cur.text_token_mapping[cur_idx];
      auto update_token_idx = update.text_token_mapping[update_idx];
      std::string cur_token_val = cur.tokens[cur_token_idx];
      std::string update_token_val = update.tokens[update_token_idx];

      printf("    CUR     text_idx->token: %d -> %d\n", cur_idx, cur_token_idx);
      printf("    CUR      cur token text: '%s'\n", cur_token_val.c_str());
      printf(" UPDATE     text_idx->token: %d -> %d\n", update_idx, update_token_idx);
      printf(" UPDATE      new token text: '%s'\n", update_token_val.c_str());
      printf("('%s'->'%s')\n", cur_token_val.c_str(), update_token_val.c_str());


      // DEBUG -- assert equal
      // std::string cur_token_str;
      // int cur_idx_copy = cur_idx;
      // while (cur.text_token_mapping[cur_idx_copy] == cur_token_idx) {
      //   cur_token_str.push_back(cur.text[cur_idx_copy++]);
      // }
      // if (cur_token_str != cur_token_val) {
      //   printf(" ERROR current string token: '%s'\n", cur_token_str.c_str());
      // }
      // Debug done 

      if ((cur_idx + cur_token_val.size()) > cur.text.size() ||
           (update_idx + update_token_val.size()) > update.text.size() ) {
        printf(" ERROR     check off by one: '%d >= %ld'\n", cur_idx + cur_token_val.size(), 
                                                                            cur.text.size());
        printf(" ERROR     check off by one: '%d >= %ld'\n", update_idx + update_token_val.size(),
                                                                            update.text.size());
      }

      // TODO:
      //   - check if equal, increase token counter
      // else
      //   - clobber and :
      //       create or search for token_confilicts
      if (cur_token_val == update_token_val) {

        // TODO:  For entire token ----
        // Increase the token reference and the count.
        // auto cur_token_idx = cur.text_token_mapping[cur_idx];
        // if (prev_token_idx != cur_token_idx) {
          // cur.counts[cur_token_idx]++;
        //   prev_token_idx = cur_token_idx;
        // }
        cur.counts[cur_token_idx]++;
        printf("EQUAL Tokens inc to: %d\n", cur.counts[cur_token_idx]);
        cur_idx += cur_token_val.size();
        update_idx += update_token_val.size();
      } else {
        // Replace the entire token with the new one
        // 1. replace text of token.
        //  1.1 if tokens same size === easy/direct replacement
        //  1.2 if old token bigger then replace and remove
        //  1.3 if new token bigger then replace and insert
        // 2. look up for matching token in conflicts lists
        //  2.1 if exists
        //   2.1.1  mark location
        //   2.1.2  else add to conflicts and mark location
        //  2.2.  Swap conflicts token with current token / probs / count.
        // 3.  Increment counters from cur_idx and update_idx

        // 1. -- 
        // auto cur_token_idx = cur.text_token_mapping[cur_idx];
        // auto update_token_idx = update.text_token_mapping[update_idx];
        if (cur_token_val.size() == update_token_val.size()) {
          // 1.1
          printf("- -1.1 same size\n");
          for (auto off = 0; off < cur_token_val.size(); off++) {
            cur.text[cur_token_idx+off] = update.text[update_token_idx+off];
          }
        } else if (cur_token_val.size() > update_token_val.size()) {
          // 1.2
          printf("- -1.2 cur token is larger\n");
          auto off = 0;
          for (; off < update_token_val.size(); off++) {
            cur.text[cur_token_idx+off] = update.text[update_token_idx+off];
          }
          cur.text.erase(cur_token_idx+off, cur_token_val.size()-off);
        } else {
          // 1.3
          printf("- -1.3 update token is larger\n");
          auto off = 0;
          for (; off < cur_token_val.size(); off++) {
            cur.text[cur_token_idx+off] = update.text[update_token_idx+off];
          }
          cur.text.insert(cur_token_idx+off, update_token_val.substr(off));
        }

        printf("- ----- REPLACED to : '%s'\n", cur.text.c_str());
        // 2. -- TODO

        // 3.
        cur_idx += cur_token_val.size();
        update_idx += update_token_val.size();
      }
    }

    // TODO:  Test removing extra from cur

    // Add what is remaining in update on to cur.text
    if (update_idx < update.text.size()) {
      cur.text += update.text.substr(update_idx);
    }
  }

  void merge(WhisperSegment & cur, WhisperSegment & update) {
    clobber(cur, update);
  }

  void clobber(WhisperSegment & cur, WhisperSegment & update) {
    printf("       update: '%s'\n", update.text.c_str());
    printf("       before: '%s'\n", cur.text.c_str());
    cur = update;
    printf("        after: '%s'\n", cur.text.c_str());
  }

  std::vector<WhisperSegment> segments;
  std::vector<bool> segments_finished;


  TranscriptionData() {};

  // void add_frame(const std::vector<std::string> & texts, 
  //         const std::vector<int64_t> & t0s, const std::vector<int64_t> & t1s) {
void add_frame(const std::vector<WhisperSegment> & add_segments) {
    // Indices of segments which are going stale
    std::vector<int> update_segments_finished;
    int earliest_updated = segments.size()+1;
    // Mask for the new segments which have been used (and should NOT be pushed to back)
    std::vector<bool> add_segments_mask(add_segments.size(), true);

    for (auto i=0; i<segments.size(); i++) {
      if (segments_finished[i]) {
        continue;
      } // If the segment does not get updated now, mark it as finished.
      bool updated = false;

      printf("old_segment: '%s'\n", segments[i].text.c_str());

      for (auto j=0; j<add_segments.size(); j++) {
        // if (!add_segments_mask[j]) {
        //   continue; // New segment has already been merged/used to update
        // }
        auto new_segment = add_segments[j];

        float overlap;
        int start_str1;
        int start_str2;
        int len;
        
        std::tie(overlap, start_str1, start_str2, len) = segments[i].text_overlap(new_segment);
        printf("   v.s. new: '%s'\n", new_segment.text.c_str());
        
        if (overlap > 0.4) {
          std::string lcs = segments[i].text.substr(start_str1, len);
          printf("\tlcs:  (%.4f % ):  '%s'\n", overlap, lcs.c_str());

          if (segments[i].text.size() <= new_segment.text.size()) {
            printf("\t!!-MERGED\n");
            merge(segments[i], new_segment, start_str1, start_str2, len);
            // segments[i] = new_segment;
            // if (!add_segments_mask[j]) {
            //   segments[i].text = "";
            // } else {
            //   segments[i] = new_segment;
            // }
          }
          add_segments_mask[j] = false; // Dont push on end of segments
          updated = true; // Dont mark as stale, or any later
          earliest_updated = std::min(earliest_updated, i); 
          break;
        }
      }

      if (!updated) {
        update_segments_finished.push_back(i);
      }
    }

    // Add new segments to transcript
    for (auto j=0; j<add_segments.size(); j++) {
      if(add_segments_mask[j]) {
        printf("Push Back: '%s'\n", add_segments[j].text.c_str());
        segments.push_back(add_segments[j]);
        segments_finished.push_back(false);
      }
    }

    // Mark segments which are going stale
    printf("///Earliest: %d\n", earliest_updated);
    for (int & i : update_segments_finished) {
      if (earliest_updated < i) {
        continue;
      }
      // if (last >= 0 && i != last+1) {
      //   break;
      // }
      printf("No longer Updated [%d]: '%s'\n", i, segments[i].text.c_str());
      segments_finished[i] = true;
    }

    // 2 _More things_
    //  - Start keeping count of overlaps.  token-word level...
    //  - lcs printout by passing start and len of calling substr.
    //  - adjust overlap.
    //  - fix merge function

    //  - Window input text
    //  - Merge special tokens (e.g. [BLANK_AUDIO])






    // 2 things
    //   1. Drop new elements from incoming array when added -- done (todo test)
    //   2. Only let last continuous indices go stale -- done (todo test)
    //   3. TODO:  Merge starting from first element of lcs (return lcs start indice, lcs len)
    //   4.  Issue:  Update is way smaller getting merged -- done

    // Next 2:
    //  1. dont remove from list
    //  2. Capitals and numbers

    // Next 1!:
    /*
    This problem is when two segments are present in the transcript that get combined by the next update from the 
      sliding window.  e.g.

    [component_container_mt-1] old_segment: ' True, Dudley was now so scared of Harry'
    [component_container_mt-1]    v.s. new: ' Harry's last month with the Dursleys wasn't fun.'
    [component_container_mt-1]    v.s. new: ' True Dudley was now so scared of Harry he wouldn't stay in the same room'
    [component_container_mt-1]  --Overlap 0.8500 
    [component_container_mt-1]  !!-MERGED
    [component_container_mt-1] old_segment: ' he wouldn't stay in the same room.'
    [component_container_mt-1]    v.s. new: ' Harry's last month with the Dursleys wasn't fun.'
    [component_container_mt-1]    v.s. new: ' True Dudley was now so scared of Harry he wouldn't stay in the same room'
    [component_container_mt-1]  --Overlap 0.9714 
    [component_container_mt-1]  !!-MERGED
      -- If used in merge, and next one overlaps too, delete the next one overlapping

    NOTE:  
      - That means we cannot get rid of v.s. new segments when they get merged, in case they are used to detect overlap
      - i.e.
[component_container_mt-1] old_segment: ' Chapter 6, The Journey from Platform 9 and 3 Quarters.'
[component_container_mt-1]    v.s. new: ' and free quarters.'
[component_container_mt-1]  --Overlap 0.5263 
[component_container_mt-1] old_segment: ' [BLANK_AUDIO]'
[component_container_mt-1]    v.s. new: ' and free quarters.'
[component_container_mt-1]    v.s. new: ' Harry's last month with the Dursleys wasn't fun.'
[component_container_mt-1]    v.s. new: ' True, Dudley was now so scared of Harry'
[component_container_mt-1]    v.s. new: ' he wouldn't stay in the same room.'
[component_container_mt-1] old_segment: ' Harry's last month with the dirt is wasn't fun.'
[component_container_mt-1]    v.s. new: ' and free quarters.'
[component_container_mt-1]    v.s. new: ' Harry's last month with the Dursleys wasn't fun.'
[component_container_mt-1]  --Overlap 0.6250 
[component_container_mt-1]  !!-MERGED
          - checks against "and free quarters" MUST STAY


    OPTIOn B:
    - 
    */
  };



  // void add_segment(const WhisperSegment & segment) {

  //   bool updated = false;

  //   // Calling back() on an empty vector is undefined
  //   if (!segments.empty()) {
  //     auto & last_segment = segments.back();
  //     float overlap = last_segment.text_overlap(segment);

  //     printf("last_segment: '%s'\n", last_segment.text.c_str());
  //     printf(" new_segment: '%s'\n", last_segment.text.c_str());
  //     printf("\t--Overlap %.4f \n", overlap);

  //     if (overlap > 0.2 && segment.text.size() >= last_segment.text.size()) {
  //       updated = true;
  //       last_segment = segment;
  //       printf("\t--MERGED\n");
  //     }
  //   }

  //   // Apply operations
  //   if (!updated) {
  //     segments.push_back(segment);
  //   }
  // };

};









class Patch {
public:
    std::vector<std::string> tokens;         // The string split into tokens
    std::vector<float> probs;                // Probability for each token
    std::vector<int> counts;                 // Count of occurrences for each token

    std::vector<std::string> tokens_match;   // Compute these for matching

    Patch(const std::vector<std::string>& tks, const std::vector<float>& probs, const std::vector<int>& counts)
        : tokens(tks), probs(probs), counts(counts) 
    {
        if (tokens.size() != probs.size() || tokens.size() != counts.size()) {
            throw std::invalid_argument("Vectors must be of the same size");
        }
    };
    Patch(const std::vector<std::string>& tks, const std::vector<float>& probs)
        : tokens(tks), probs(probs)
    {
        counts = std::vector<int>(tokens.size(), 1);
        if (tokens.size() != probs.size() || tokens.size() != counts.size()) {
            throw std::invalid_argument("Vectors must be of the same size");
        }
    };
    Patch(){};


    std::string get() const {
        std::string ret = "";
        for (auto& token : tokens) {
            ret += token;
        }
        return ret;
    }

    std::vector<std::string> get_matchable(const int &first_token) {
        // Return match-able tokens.
        // All lower-case
        // Whitespaces removed 
        while (tokens_match.size() < tokens.size()) {
            auto conversion = normalize_string(tokens[tokens_match.size()]);
            tokens_match.push_back(conversion);
        }

        std::vector<std::string> ret_splice;
        auto cur = first_token;
        while (cur < tokens_match.size()) {
            ret_splice.push_back(tokens_match[cur++]);
        }
        return ret_splice;
    }

    std::vector<std::string> create_matchable() const {
        // Does not update the "tokens_match" field
        //  Use this so we can keep const qualiffier and use on update object
        std::vector<std::string> ret;
        auto cur = 0;
        while (cur < tokens.size()) {
            auto conversion = normalize_string(tokens[cur]);
            ret.push_back(conversion);
            cur++;
        }
        return ret;
    }


    inline const std::size_t &size() const { return tokens.size(); };

    // Helper string operations
    std::string normalize_string(const std::string& str) const {
        std::string result;
        
        // Convert to lowercase and remove whitespace in one pass
        for (char ch : str) {
            if (!std::isspace(static_cast<unsigned char>(ch))) { // skip whitespace
                result += std::tolower(static_cast<unsigned char>(ch));
            }
        }

        return result;
    }

};

class PatchOperation {
public:
    enum Type { INSERT, PUSH, REPLACE, DELETE } type;
    Patch update;
    int index; // Index at which the operation applies

    // For reversal, we store the inverse operation
    PatchOperation reverse(const Patch& target) const {
        switch (type) {
            case INSERT:
                return { DELETE, Patch(update.tokens, update.probs, update.counts), index };
            case PUSH: {
                int pushIndex = target.tokens.size() - update.tokens.size();
                return { DELETE, Patch(update.tokens, update.probs, update.counts), pushIndex };
            }
            case REPLACE: {
                // Capture the original tokens, probabilities, and counts for reversal
                auto original_tokens = std::vector<std::string>(target.tokens.begin() + index, target.tokens.begin() + index + update.tokens.size());
                auto original_probs = std::vector<float>(target.probs.begin() + index, target.probs.begin() + index + update.probs.size());
                auto original_counts = std::vector<int>(target.counts.begin() + index, target.counts.begin() + index + update.counts.size());
                
                return { REPLACE, Patch(original_tokens, original_probs, original_counts), index };
            }
            case DELETE:
                return { INSERT, Patch(update.tokens, update.probs, update.counts), index };
        }
        throw std::logic_error("Unknown operation type");
    }
};


class TranscriptData {
public:
    Patch transcript;
    std::vector<Patch> transcript_frames;
    std::vector<std::vector<PatchOperation>> ops;

    // Parameters
    int allowed_substr_gaps;

    // 
    // Core algorithms
    // 
    
    int edit_distance(const std::string& s1, const std::string& s2);
    
    // transcript[match_id.first] ~= update.tokens[match_id.second]
    // void lcs(const Patch update, std::vector<std::pair<size_t, size_t>> & match_ids);
    
    // Returns: start, length
    // std::pair<size_t, size_t> find_longest_substr(const std::vector<std::pair<size_t, size_t>> & match_ids);

    // void lcs_merge(const Patch update);


    // Helper
    int last_update_token;
    // std::vector<Patch> splice(const int start, const int length) {
    //     std::vector<Patch> ret;
    //     for (int i = start; i < start + length; ++i) {
    //         ret.push_back(transcript_frames[i]);
    //     }
    //     return ret;
    // }

    std::string get_splice(const std::string t, const int start, const int length) {
        std::string ret;
        for (int i = start; (i < start + length && i < t.size()); i++) {
            ret.push_back(t[i]);
        }
        return ret;
    };


    std::vector<PatchOperation> lcs_merge(const Patch update) {
        // auto frame_idx = transcript_frames.size() - last_updated_frame;
        // if ( frame_idx >= 0 ) {
        //     auto start = transcript_frames.size()-1-last_updated_frame;
        //     auto length = transcript_frames.size() - start;
        //     auto & patches = splice(start, length);
        // } else {
        //     auto & patches = transcript_frames;
        // }

        // if (patches.size() == 0) {
        //     transcript = update;
        //     return;
        // }
        std::vector<PatchOperation> update_ops;

        std::string text = transcript.get();
        // printf("Text: %s\n", text.c_str());
        if (text.size() == 0) {
            last_update_token = 0;
            update_ops.push_back({PatchOperation::PUSH, update, -1});
            return update_ops;
        }
        // int start = transcript.size() - 1 - last_update_token;
        // int start = last_update_token;
        // int length = text.size() - start;

        // std::string splice;
        // if (start > 0) {
        //     printf("Splice: start, end, length, size: [%d, %d), %d | %ld\n", 
        //                                 start, length+start, length, text.size());
        //     splice = get_splice(text, start, length);
        // } else {
        //     // Use entire text
        //     splice = text;
        // }

        printf("Transcript full: %s\n", text.c_str());
        // printf("Transcript trim: %s\n", splice.c_str());
        printf("Update: %s\n", update.get().c_str());

        std::vector<std::pair<int, int>> match_ids;
        lcs(update, match_ids);
        // Print matches
        std::string matched_print = "";
        printf("\tMatched token pairs: ");
        for (auto [i, j] : match_ids) {
            printf("(%d,%d) ", i, j);
            matched_print += update.tokens[j] + " ";
        }
        printf("\n");
        printf("\tMatched: %s\n", matched_print.c_str());

        if (match_ids.size() == 0) {
            last_update_token = 0; // TODO
            update_ops.push_back({PatchOperation::PUSH, update, -1});
            return update_ops;
        }

        
        std::vector<std::string> matched_tokens;
        std::vector<float> probs;


        update_ops.push_back({PatchOperation::PUSH, update, -1});
        auto transcript_idx = match_ids[0].first;
        auto update_idx = match_ids[0].second;
        for (auto [match_transcript_id, match_update_id] : match_ids) {
            // TODO
        }

        /*
        std::vector<std::pair<size_t, size_t>> match_ids;
        lcs(update, match_ids);
        if (match_ids.size() == 0) {
            // Add update to end
            last_update_token = 0; // TODO last_update_token = transcript.size()-1;
            update_ops.push_back({PatchOperation::PUSH, update, -1});
            return update_ops;
        }

        // Print matches
        std::string matched_print = "";
        printf("\tMatched token pairs: ");
        for (auto [i, j] : match_ids) {
            printf("(%ld,%ld) ", i, j);
            matched_print += update.tokens[j] + " ";
        }
        printf("\n");
        printf("\tMatched: %s\n", matched_print.c_str());
        */


        // Find the longest common substring
        // find_longest_substr(match_ids);

        // Assert that we have found at least 5 tokens that match
        // if (match_ids.size() < 6) {
        //     last_update_token = 0; // TODO
        //     update_ops.push_back({PatchOperation::PUSH, update, -1});
        //     return update_ops;
        // }




        return update_ops;


        // auto [start, length] = find_longest_substr(match_ids);
        // if (length == 0) {
        //     transcript = update;
        //     return ;
        // } 
        // fprintf(stderr, "%s: not running!\n", __func__);


        // std::vector<PatchOperation> frame_ops;
        // for (auto [i, j] : match_ids) {


        //     if (i < start || i >= start + length) {
        //         frame_ops.push_back({PatchOperation::REPLACE, Patch({update.tokens[j]}, {update.probs[j]}), i});
        //     }
        // }
        // PatchOperation op = { PatchOperation::REPLACE, Patch(update.tokens, update.probs), start };
        // ops.push_back({op});
        // applyPatchOperation(transcript, op);
    };

    TranscriptData(){
        last_update_token = 0;
        allowed_substr_gaps = 5;
    };

    // Add the update to the transcript_frames
    void push_frame(const std::vector<std::string>& texts, const std::vector<float>& probs) {
        // debugging:
        // printf("Starging Merge\n");
        // auto patch = Patch(texts, probs);
        // auto frame_ops = lcs_merge(patch);
        // apply(patch, frame_ops);
        transcript_frames.push_back({texts, probs});



        // workds:
        // transcript_frames.push_back(Patch(texts, probs));
        // auto& update = transcript_frames[transcript_frames.size()-1];
        // make_merge(update);
    };

    void preprocess(std::vector<std::string>& texts, std::vector<float>& probs) {
        // TODO:  Verify operation (and average of probabilities)
        // Preprocess the text tokens, 
        const std::vector<std::pair<std::string, std::string>> bracket_pairs = {
            {"[", "]"}, {" [", "]"},
            {"{", "}"}, {" {", "}"},
            {"(", ")"}, {" (", ")"}
        };
        const size_t max_allowed_tokens_to_combine = 8;

        // If it is to be removed, the two indicies will be the same.
        //   Otherwise combine such that [start, end)
        std::vector<std::pair<size_t, size_t>> combine_remove;
        for (size_t i = 0; i < texts.size(); i++) {
            // If the text starts with "[_" and ends with "]", 
            //       then it is a whisper specific token so skip it
            if (texts[i].substr(0, 2) == "[_" && texts[i].substr(texts[i].size() - 1, 1) == "]") {
                combine_remove.push_back({i, i });
                continue;
            }

            // Search for a matching bracket
            for (auto &[start, end] : bracket_pairs) {
                if (texts[i].size() < start.size()) {
                    continue;
                }
                if (texts[i].substr(0, start.size()) == start ) {
                    bool end_found = false;
                    size_t end_idx = i+1;
                    while (end_idx < texts.size() && (end_idx-i) <= max_allowed_tokens_to_combine) {
                        if (texts[end_idx].substr(0, end.size()) == end) {
                            end_found = true;
                            break;
                        }
                        end_idx++;
                    }
                    if (end_found) {
                        // Combine such that:   [start, end) 
                        combine_remove.push_back({i, end_idx+1});
                        i = end_idx;
                    }
                }
            }
        }

        // Apply drop/combine opperations
        int offset = 0; // increment offest for every removed element
        for (auto &[start, end] : combine_remove) {

            if (start == end) {
                // printf("Removing:  %s\n", texts[start-offset].c_str());
                texts.erase(texts.begin() + start - offset);
                probs.erase(probs.begin() + start - offset);
                offset++;
                continue;
            }

            // printf("Combining (with %s):  ", texts[start - offset].c_str());
            auto cur = start + 1;
            while (cur != end) {
                // printf("%s ", texts[cur - offset].c_str());
                texts[start - offset] += texts[cur - offset];
                probs[start - offset] += probs[cur - offset];
                cur++;
            }
            // printf("\n");
            // Erase values from arrray
            texts.erase(texts.begin() + start + 1 - offset, texts.begin() + end - offset);
            probs.erase(probs.begin() + start + 1 - offset, probs.begin() + end - offset);
            // Average probabilities
            probs[start - offset] /= (end - start);
            offset += (end - start - 1);
        }
    };

    void apply(const Patch& update, const std::vector<PatchOperation>& ops) {
        // Apply the operations to the transcript
        for (auto& op : ops) {
            applyPatchOperation(transcript, op);
        }
        // Keep for records
        transcript_frames.push_back(update);
    };

    std::string get_last() {
        if (transcript_frames.size() == 0) {
            return "";
        }
        return transcript_frames[transcript_frames.size()-1].get();
    };

    std::string get() const {
        return transcript.get();
        if (transcript_frames.size() == 0) {
            return "";
        }
        std::string ret = "";
        for (auto& patch : transcript_frames) {
            ret += patch.get();
        }
        return ret;
    };

    void clear() {
        transcript_frames.clear();
    };

    void applyPatchOperation(Patch& target, const PatchOperation& op) {
        switch (op.type) {
            case PatchOperation::INSERT:
                target.tokens.insert(target.tokens.begin() + op.index, op.update.tokens.begin(), op.update.tokens.end());
                target.probs.insert(target.probs.begin() + op.index, op.update.probs.begin(), op.update.probs.end());
                target.counts.insert(target.counts.begin() + op.index, op.update.counts.begin(), op.update.counts.end());
                break;
            case PatchOperation::PUSH:
                target.tokens.insert(target.tokens.end(), op.update.tokens.begin(), op.update.tokens.end());
                target.probs.insert(target.probs.end(), op.update.probs.begin(), op.update.probs.end());
                target.counts.insert(target.counts.end(), op.update.counts.begin(), op.update.counts.end());
                break;
            case PatchOperation::REPLACE:
                if (op.index + op.update.tokens.size() > target.tokens.size()) {
                    throw std::out_of_range("Replace operation out of range");
                }
                for (size_t i = 0; i < op.update.tokens.size(); ++i) {
                    target.tokens[op.index + i] = op.update.tokens[i];
                    target.probs[op.index + i] = op.update.probs[i];
                    target.counts[op.index + i] = op.update.counts[i];
                }
                break;
            case PatchOperation::DELETE:
                if (op.index + op.update.tokens.size() > target.tokens.size()) {
                    throw std::out_of_range("Delete operation out of range");
                }
                target.tokens.erase(target.tokens.begin() + op.index, target.tokens.begin() + op.index + op.update.tokens.size());
                target.probs.erase(target.probs.begin() + op.index, target.probs.begin() + op.index + op.update.probs.size());
                target.counts.erase(target.counts.begin() + op.index, target.counts.begin() + op.index + op.update.counts.size());
                break;
        }
    };

    void reversePatchOperations(Patch& target, const std::vector<PatchOperation>& ops) {
        for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
            applyPatchOperation(target, it->reverse(target));
        }
    };

    void lcs(const Patch update, std::vector<std::pair<int, int>> & match_ids) {
        const int maxGap = 10;
        auto v1 = transcript.get_matchable(last_update_token);
        auto v2 = update.create_matchable();

        /* print two strings */
        // std::string s1 = "";
        // for (auto yy : v1) {
        //     s1 += yy + " ";
        // }
        // std::string s2 = "";
        // for (auto yy : v2) {
        //     s2 += yy + " ";
        // }
        // printf("\tLast Update Token:  %d\n", last_update_token);
        // printf("\ttranscript_match:  %s\n", s1.c_str());
        // printf("\tupdate_match:  %s\n", s2.c_str());

        int n1 = v1.size();
        int n2 = v2.size();
        // dp[i][j] will store the length of the longest subsequence ending at v1[i] and v2[j]
        std::vector<std::vector<int>> dp(n1 + 1, std::vector<int>(n2 + 1, 0));
        // prev[i][j] stores the previous index pair that contributes to the current dp[i][j]
        std::vector<std::vector<std::pair<int, int>>> prev(n1 + 1, 
                            std::vector<std::pair<int, int>>(n2 + 1, {-1, -1}));

        // Track the maximum length and the end indices of that subsequence
        int maxLen = 0;
        std::pair<int, int> maxEnd = {-1, -1};

        // Traverse both vectors in reverse order to perform backward search
        for (int i = n1 - 1; i >= 0; --i) {
            for (int j = n2 - 1; j >= 0; --j) {
                if (v1[i] == v2[j]) {
                    // Try to extend the match from current indices (i, j)
                    dp[i][j] = 1; // At least the current element matches
                    prev[i][j] = {-1, -1}; // Default previous index (no continuation yet)

                    // Check all possible continuations within the gap allowance
                    for (int gap1 = 1; gap1 <= maxGap && i + gap1 < n1; ++gap1) {
                        for (int gap2 = 1; gap2 <= maxGap && j + gap2 < n2; ++gap2) {
                            if (dp[i + gap1][j + gap2] + 1 > dp[i][j]) {
                                dp[i][j] = dp[i + gap1][j + gap2] + 1;
                                prev[i][j] = {i + gap1, j + gap2};
                            }
                        }
                    }

                    // Track the maximum sequence found
                    if (dp[i][j] > maxLen) {
                        maxLen = dp[i][j];
                        maxEnd = {i, j};
                    }
                }
            }
        }


        // print dp table
        // for (size_t i=0; i<dp.size(); i++) {
        //     for (size_t j=0; j<dp[i].size(); j++) {
        //         printf("%d ", dp[i][j]);
        //     }
        //     printf("\n");
        // }

        // Reconstruct the sequence of matches
        for (auto [i, j] = maxEnd; i != -1 && j != -1; std::tie(i, j) = prev[i][j]) {
            match_ids.emplace_back(i, j);
        }

        // Since we built the sequence in reverse order, reverse it back
        std::reverse(match_ids.begin(), match_ids.end());
    }



    void lcs_old(const Patch update, std::vector<std::pair<size_t, size_t>> & match_ids) {
        // Find the common substrings between trancript and update.
        // Postprocess to large gaps and keep a continuous substring.
        // Return longest common substring and indexes
        auto s_update = update.create_matchable();
        auto s_transcript = transcript.get_matchable(last_update_token);
        size_t m = s_transcript.size();
        size_t n = s_update.size();

        /* print two strings */
        std::string s1 = "";
        for (auto yy : s_transcript) {
            s1 += yy + " ";
        }
        std::string s2 = "";
        for (auto yy : s_update) {
            s2 += yy + " ";
        }
        // printf("\tLast Update Token:  %d\n", last_update_token);
        // printf("\ttranscript_match:  %s\n", s1.c_str());
        // printf("\tupdate_match:  %s\n", s2.c_str());
        
        std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
        for (size_t i = 1; i <= m; ++i) {
            for (size_t j = 1; j <= n; ++j) {
                // auto min_len = std::min(s_transcript[i - 1].size(), s_update[j - 1].size());
                if (s_transcript[i - 1] == s_update[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }
                // else if (min_len >= allowed_str_min_len && edit_distance(transcript[i - 1], update[j - 1]) <= allowed_str_edit) {
                //     dp[i][j] = dp[i - 1][j - 1] + 1;
                // } 
                else {
                    dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        // print dp table
        // for (size_t i=0; i<dp.size(); i++) {
        //     for (size_t j=0; j<dp[i].size(); j++) {
        //         printf("%d ", dp[i][j]);
        //     }
        //     printf("\n");
        // }

        if (dp[m][n] == 0) {
            // No matches
            return;
        }

        // Reconstruct the matching substrings and save their indexes.
        //   Offest indicies with last update token
        int i = m, j = n;
        while (i > 0 && j > 0) {
            if (dp[i][j] > std::max(dp[i - 1][j], dp[i][j - 1])) {
                match_ids.push_back({i+last_update_token-1, j-1});
                --i;
                --j;
            } else if (dp[i - 1][j - 1] == dp[i][j]) {
                --i;
                --j;
            } else if (dp[i - 1][j] > dp[i][j - 1]) {
                --i;
            } else {
                --j;
            }
        }

        // Reconstruction was done in reverse, fix the indexes
        // std::reverse(match_ids.begin(), match_ids.end());
    };

    void find_longest_substr(std::vector<std::pair<size_t, size_t>> & match_ids) {
        int substr_len = 1;
        int len = 1;
        int start = 0;

        for (int i=1; i<match_ids.size(); i++) {
            auto update_gap = std::abs(static_cast<int>(match_ids[i].second) - 
                                            static_cast<int>(match_ids[i-1].second));
            auto transcript_gap = std::abs(static_cast<int>(match_ids[i].first) - 
                                            static_cast<int>(match_ids[i-1].first));
            if (update_gap < allowed_substr_gaps && transcript_gap < allowed_substr_gaps) {
                substr_len++;
            } else {
                if (len <= substr_len) {
                    len = substr_len;
                    start = i - len;
                }
                substr_len = 1;
            }
        }

        //  using "<=" means Keep the right-most longest substring.
        if (len <= substr_len) {
            len = substr_len;
            start = match_ids.size() - len;
        }


        // Keep match_ids for only the longest substring.  
        //    Remove end first so idxs are still good
        printf("\tBefore: ");
        for (auto [i, j] : match_ids) {
            printf("(%ld,%ld) ", i, j);
        }
        printf("\n");
        match_ids.erase(match_ids.begin() + start + len , match_ids.end());
        match_ids.erase(match_ids.begin(), match_ids.begin() + start);
        printf("\tAfter: ");
        for (auto [i, j] : match_ids) {
            printf("(%ld,%ld) ", i, j);
        }
        printf("\n");
    };






//     // Do post-processing on reversed arrays, push into output backwards
//     // std::reverse(lcs_.begi(), lcs_.end());
//     // std::reverse(lcs_ids_transcript_.begi(), lcs_ids_transcript_.end());
//     // std::reverse(lcs_ids_update_.begi(), lcs_ids_update_.end());

//     // Post-Process
//     auto result = find_longest_substr(lcs_ids_update_, lcs_ids_transcript_, allowed_substr_gaps);
//     // print_debug("\tLCS Start: " + std::to_string(result.second));
//     // print_debug("\tLCS Length: " + std::to_string(result.first));
    
//     // Insert values into lcs/lcs_ids_transcript/lcs_ids_update. 
//     //  Do this backwards! since the reconstruction was in reversed order!
//     for (auto i=result.second+result.first-1; i>=result.second; i--) {
//         lcs.push_back(lcs_[i]);
//         lcs_ids_transcript.push_back(lcs_ids_transcript_[i]);
//         lcs_ids_update.push_back(lcs_ids_update_[i]);
//     }
//     return;
// }

































//     std::vector<std::vector<std::string>> p_frames;
//     std::vector<std::string> transcript;
//     std::vector<float> p;
//     std::vector<int> token_count;

//     TranscriptData(const std::vector<std::string>& transcript = {},
//                    const std::vector<float>& transcript_p = {},
//                    const std::vector<int>& transcript_token_count = {})
//         : transcript(transcript), p(transcript_p), 
//             token_count(std::vector<int>(transcript.size(), 1)) {}


//     void append(const TranscriptData& update, const size_t count, const int update_start=0);
//     void replace(const TranscriptData& update, const size_t count, 
//                         const int update_start, const int transcript_start);
//     void insert(const TranscriptData& update, const size_t count, 
//                         const int update_start, const int transcript_start);
//     void remove(size_t index, size_t count=1);

//     void update_word(const int idx, const float prob);
//     void replace_word(const int idx, const std::string word, const float prob);
//     void update_replace_word(const int idx, const TranscriptData& update, const int update_idx);

//     float get_prob(const int start, const int end) const;
//     float get_prob(const int idx) const {
//         return p[idx];
//     }

//     // void remove_if(const int idx);
//     void append(const std::string token, const float prob) {
//         transcript.push_back(token);
//         p.push_back(prob);
//         token_count.push_back(1);
//     }

//     std::vector<std::string> splice(const int start, const int count) const;
    
//     size_t size() const {
//         return transcript.size();
//     }

//     // std::vector<std::string> push_frame(std::vector<std::string>& texts, std::vector<float> probs);

};

} // end of namespace whisper
#endif // WHISPER_UTIL__TRANSCRIPT_DATA_HPP_
