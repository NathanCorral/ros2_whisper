#include "whisper_util/transcript_updater.hpp"

namespace whisper {
    
// Print debug message if DEBUG_MODE is enabled
void TranscriptUpdater::print_debug(const std::string& message, const bool new_line) const {
    if (DEBUG_MODE) {
        std::cout << message;
        if (new_line){
            std::cout << std::endl;
        }
    }
}

// Debug helper to print vectors
void TranscriptUpdater::print_vector_debug(const std::vector<std::string>& vec, const bool new_line) const {
    if (DEBUG_MODE) {
        for (const auto& s : vec) {
            std::cout << s << " ";
        }
        if (new_line){
            std::cout << std::endl;
        }
    }
}

void TranscriptUpdater::print_vector_debug(const std::vector<int>& vec, const bool new_line) const {
    if (DEBUG_MODE) {
        for (const auto& s : vec) {
            std::cout << s << " ";
        }
        if (new_line){
            std::cout << std::endl;
        }
    }
}

void TranscriptUpdater::print_dp_table_debug(const std::vector<std::vector<int>>& dp) const {
    if (DEBUG_MODE) {
        for (const auto& row : dp) {
            for (const auto& val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }
}

std::string TranscriptUpdater::replaceNumberWordsWithDigits(const std::string &input) {
    std::string result = input;
    
    // Use a regular expression to match word-based numbers in a case-insensitive way
    for (const auto& pair : number_map) {
        std::regex word_regex("\\b" + pair.first + "\\b", std::regex_constants::icase);
        result = std::regex_replace(result, word_regex, pair.second);
    }
    
    return result;
}

// Convert number words to digits
std::string TranscriptUpdater::matchingTransform(const std::string &input) {
    // return normalizeString(input);
    // return replacePunctuation(replaceNumberWordsWithDigits(normalizeString(input)));
    return replacePunctuation(normalizeString(input));
}

// Convert digit numbers to their word equivalents
std::string TranscriptUpdater::replaceDigitsWithNumberWords(const std::string &input) {
    std::string result = input;

    // Use a regular expression to find digits
    for (const auto& pair : number_map) {
        std::regex digit_regex("\\b" + pair.second + "\\b");
        result = std::regex_replace(result, digit_regex, pair.first);
    }
    
    return result;
}

std::string TranscriptUpdater::normalizeString(const std::string& str) {
    std::string result;
    
    // Convert to lowercase and remove whitespace in one pass
    for (char ch : str) {
        if (!std::isspace(static_cast<unsigned char>(ch))) { // skip whitespace
            result += std::tolower(static_cast<unsigned char>(ch));
        }
    }

    return result;
}

std::string TranscriptUpdater::replacePunctuation(const std::string& input) {
    if (std::find(punctuation_map.begin(), punctuation_map.end(), input) 
                                                     != punctuation_map.end()) {
        // Doesnt really matter what we return, just that they all get set to the
        //  same token/string for matching purposes
        return "-";
    }
    return input; // else do nothing
}


// Helper function to compute edit distance (used for fuzzy matching)
int edit_distance(const std::string& s1, const std::string& s2) {
    int m = s1.size();
    int n = s2.size();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));

    for (int i = 0; i <= m; ++i) dp[i][0] = i;
    for (int j = 0; j <= n; ++j) dp[0][j] = j;

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = std::min({dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1});
            }
        }
    }
    return dp[m][n];
}

std::pair<int, int> find_longest_substr(const std::vector<int>& lcs_ids_update,
                                        const std::vector<int>& lcs_ids_transcript,
                                        const int allowed_substr_gaps);


void TranscriptUpdater::find_lcs(const std::vector<std::string>& transcript,
                            const std::vector<std::string>& update,
                            std::vector<std::string>& lcs,
                            std::vector<int>& lcs_ids_transcript,
                            std::vector<int>& lcs_ids_update) {
    // Find the common substrings between trancript and update.
    // Postprocess to large gaps and keep a continuous substring.
    // Return longest common substring and indexes
    size_t m = transcript.size();
    size_t n = update.size();

    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            auto min_len = std::min(transcript[i - 1].size(), update[j - 1].size());
            if (transcript[i - 1] == update[j - 1]) {
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

    if (dp[m][n] == 0) {
        // No matches
        return;
    }

    // Reconstruct the matching substrings and save their indexes
    std::vector<std::string> lcs_;
    std::vector<int> lcs_ids_transcript_;
    std::vector<int> lcs_ids_update_;

    int i = m, j = n;
    while (i > 0 && j > 0) {
        if (dp[i][j] > std::max(dp[i - 1][j], dp[i][j - 1])) {
            lcs_.push_back(transcript[i - 1]);
            lcs_ids_transcript_.push_back(i - 1);
            lcs_ids_update_.push_back(j - 1);
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

    // Do post-processing on reversed arrays, push into output backwards
    // std::reverse(lcs_.begi(), lcs_.end());
    // std::reverse(lcs_ids_transcript_.begi(), lcs_ids_transcript_.end());
    // std::reverse(lcs_ids_update_.begi(), lcs_ids_update_.end());

    // Post-Process
    auto result = find_longest_substr(lcs_ids_update_, lcs_ids_transcript_, allowed_substr_gaps);
    // print_debug("\tLCS Start: " + std::to_string(result.second));
    // print_debug("\tLCS Length: " + std::to_string(result.first));
    
    // Insert values into lcs/lcs_ids_transcript/lcs_ids_update. 
    //  Do this backwards! since the reconstruction was in reversed order!
    for (auto i=result.second+result.first-1; i>=result.second; i--) {
        lcs.push_back(lcs_[i]);
        lcs_ids_transcript.push_back(lcs_ids_transcript_[i]);
        lcs_ids_update.push_back(lcs_ids_update_[i]);
    }
    return;
}

void TranscriptUpdater::make_merge(TranscriptData& transcript,
                    const TranscriptData & update,
                    const std::vector<int>& lcs_ids_transcript,
                    const std::vector<int>& lcs_ids_update) {

    // Throw error if lcs_ids_transcript.size() != lcs_ids_update.size()
    if (lcs_ids_transcript.size() != lcs_ids_update.size()) {
        throw std::runtime_error("ERROR:  TranscriptUpdater::make_merge() - " 
                                 "lcs_ids_transcript.size() != lcs_ids_update.size()");
    }

    if (lcs_ids_transcript.size() == 0) {
        transcript.append(update, update.transcript.size());
        return;
    }

    auto transcript_idx_offset = 0; // Handle deletions/insertions in transcript
    auto update_idx = lcs_ids_update[0];
    auto transcript_idx = lcs_ids_transcript[0];
    for (int i=0; i<lcs_ids_update.size(); i++, update_idx++, transcript_idx++) {
        /**
         * There are possibly:
         *   1. Different words in the update/transcript
         *   2. Missing words in the update
         *   3. Missing words in the transcript
         *   4. The update/transcript words match exactly
         *      - Will always take place when the indexes match the lcs substring
        **/
        if (update_idx != lcs_ids_update[i] && transcript_idx != lcs_ids_transcript[i]) {
            // Case 1:  There are differences in the udpate/transcript subsrings
            // Count the number of un-matched words in each substrings
            auto update_count = lcs_ids_update[i] - update_idx;
            auto transcript_count = lcs_ids_transcript[i] - transcript_idx;

            // Accumulate the probabilities of the differing substrings
            float update_prb = update.get_prob(update_idx, lcs_ids_update[i]);
            auto transcript_prb = transcript.get_prob(transcript_idx + transcript_idx_offset,
                                                    lcs_ids_transcript[i] + transcript_idx_offset);
            if (DEBUG_MODE) {
                auto idx = transcript_idx + transcript_idx_offset;
                std::vector<std::string> splice_transcript(transcript.transcript.begin() + idx, 
                                            transcript.transcript.begin() + idx + transcript_count);
                std::vector<std::string> splice_update(update.transcript.begin() + update_idx, 
                                            update.transcript.begin() + update_idx + update_count);
                std::cout << "\tCompare ("  << transcript_prb << "):  '";
                print_vector_debug(splice_transcript, false);
                std::cout << "' with (" << update_prb << "):  '";
                print_vector_debug(splice_update, false);
                std::cout << "'" << std::endl;
                if(update_prb > transcript_prb) {
                    print_debug("\t\tREPLACED!");
                } 
            }

            if (update_prb > transcript_prb) {
                auto replace_elements = std::min(update_count, transcript_count);
                auto insert_elements = update_count - replace_elements;
                auto remove_elements = transcript_count - replace_elements;

                transcript.replace(update, replace_elements, update_idx, 
                                    transcript_idx + transcript_idx_offset);
                update_idx += replace_elements;
                transcript_idx += replace_elements;

                transcript.insert(update, insert_elements, update_idx, 
                                    transcript_idx + transcript_idx_offset);
                update_idx += insert_elements;
                transcript_idx_offset += insert_elements;  // Increment offset for insertions

                transcript.remove(transcript_idx + transcript_idx_offset, remove_elements);
                transcript_idx_offset -= remove_elements;  // Decrement offset for insertions
                transcript_idx += remove_elements;         // Skip removed elements
            } else {
                // Reduce probability of elements in transcript
                for(auto j=0; j<transcript_count; j++) {
                    auto idx = transcript_idx + j + transcript_idx_offset;
                    transcript.p[idx] /= 2.0;
                }

                update_idx = lcs_ids_update[i];
                transcript_idx = lcs_ids_transcript[i];
            }
        }
        else if (update_idx == lcs_ids_update[i] && transcript_idx != lcs_ids_transcript[i]) {
            // Case 2:  Missting words in the update
            if (DEBUG_MODE) {
                auto idx = transcript_idx + transcript_idx_offset;
                auto transcript_count = lcs_ids_transcript[i] - transcript_idx;
                auto transcript_prb = transcript.get_prob(transcript_idx + transcript_idx_offset,
                                                    lcs_ids_transcript[i] + transcript_idx_offset);
                auto splice_transcript = transcript.splice(idx, transcript_count);
                std::cout << "\tConsider Removing ("  << transcript_prb << "):  '";
                print_vector_debug(splice_transcript, false);
                std::cout << "'" << std::endl;
            }
            // Remove entire group with lower probabiltiy
            auto transcript_prb = transcript.get_prob(transcript_idx + transcript_idx_offset,
                                                lcs_ids_transcript[i] + transcript_idx_offset);
            if (transcript_prb < 0.5) {
                auto remove_elements = lcs_ids_transcript[i] - transcript_idx;
                transcript.remove(transcript_idx + transcript_idx_offset, remove_elements);
                transcript_idx_offset -= remove_elements;  // Decrement offset for insertions
                transcript_idx += remove_elements;         // Skip removed elements
            }

            // And/Or Remove words with lower probability
            for (auto j=transcript_idx; j<lcs_ids_transcript[i]; j++) {
                auto idx = j + transcript_idx_offset;
                if (transcript.get_prob(j) < 0.5) {
                    print_debug("\t\tRemoved:  " + transcript.transcript[idx]);
                    transcript.remove(idx);
                    transcript_idx_offset--;
                } else {
                    // Reduce probability
                    transcript.p[idx] /= 2.0;
                }
            }
            transcript_idx = lcs_ids_transcript[i];
        }
        else if (update_idx != lcs_ids_update[i] && transcript_idx == lcs_ids_transcript[i]) {
            // Case 3: Missing words in the transcript
            // Insert the words for now.   
            // TODO maybe check probability or insert with lower probability
            // TODO maybe skip first/last position (often words from incomplete audio)
            transcript.insert(update, lcs_ids_update[i] - update_idx, update_idx, 
                        transcript_idx + transcript_idx_offset);
            transcript_idx_offset += lcs_ids_update[i] - update_idx;
            update_idx = lcs_ids_update[i];
        }

        // Case 4: Now we have a guaranteed strong match, i.e. 
        //    update_idx == lcs_ids_update[i] && transcript_idx == lcs_ids_transcript[i]
        transcript.update_replace_word(transcript_idx + transcript_idx_offset, update, update_idx);
    }

    // Any additional update should be added on the end of transcript
    transcript.append(update, update.size()-update_idx, update_idx);
}


// Main function that merges updates into the transcript
int TranscriptUpdater::lcs_merge(TranscriptData& transcript,
                                const TranscriptData& update,
                                const int transcript_start) {

    // Reconstruct the matching substrings and save their indexes
    std::vector<std::string> lcs;
    std::vector<int> lcs_ids_transcript;
    std::vector<int> lcs_ids_update;

    // Create a splice of the transcript using the start index
    std::vector<std::string> splice_transcript(transcript.transcript.begin() + transcript_start, 
                                                transcript.transcript.end());
    // Use a copy of update as well and apply normalization for easier string matching
    std::vector<std::string> normalized_update(update.transcript);
    for (auto& word : normalized_update) {
        word = matchingTransform(word);
    }
    for (auto& word : splice_transcript) {
        word = matchingTransform(word);
    }
    print_debug("Transcript: ", false);
    print_vector_debug(splice_transcript);
    print_debug("Update: ", false);
    print_vector_debug(normalized_update);
    find_lcs(splice_transcript, normalized_update, lcs, lcs_ids_transcript, lcs_ids_update);
    
    // Add the transcript_start to ever index in lcs_ids_transcript
    for (auto& idx : lcs_ids_transcript) {
        idx += transcript_start;
    }

    print_debug("\tLCS result: ", false);
    print_vector_debug(lcs);
    print_debug("\tLCS Transcript IDX's: ", false);
    print_vector_debug(lcs_ids_transcript);
    print_debug("\tLCS Update IDX's: ", false);
    print_vector_debug(lcs_ids_update);
    print_debug("\tLCS Length: " + std::to_string(lcs.size()));



    make_merge(transcript, update, lcs_ids_transcript, lcs_ids_update);

    print_debug("RESULTS:  ");
    print_vector_debug(transcript.transcript);
    return lcs_ids_transcript.size() == 0 ? 0 : lcs_ids_transcript[0];
}

// Utility function to find longest matching substrings (helper function)
std::pair<int, int> find_longest_substr(const std::vector<int>& lcs_ids_update,
                                        const std::vector<int>& lcs_ids_transcript,
                                        const int allowed_substr_gaps) {
    auto longest_substr = 1;
    auto longest_substr_best = 1;
    auto longest_substr_start = 0;

    for (size_t i = 0; i < (lcs_ids_transcript.size() - 1); i++) {
        auto update_gap = std::abs(lcs_ids_update[i + 1] - lcs_ids_update[i]);
        auto transcript_gap = std::abs(lcs_ids_transcript[i + 1] - lcs_ids_transcript[i]);

        if (update_gap < allowed_substr_gaps && transcript_gap < allowed_substr_gaps) {
            longest_substr++;
        } else {
            if (longest_substr_best <= longest_substr) {
                longest_substr_best = longest_substr;
                longest_substr_start = i - (longest_substr_best - 1);
            }
            longest_substr = 1;
        }
    }
    // Compare less than equal here and above because we are comparing the strings in reverse.
    //  =Keep the left-most longest substring.
    if (longest_substr_best <= longest_substr) {
        longest_substr_best = longest_substr;
        longest_substr_start = lcs_ids_transcript.size() - longest_substr_best;
    }

    return {longest_substr_best, longest_substr_start};
}
} // end of namespace whisper
