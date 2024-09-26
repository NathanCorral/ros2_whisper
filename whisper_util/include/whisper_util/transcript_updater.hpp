#ifndef WHISPER_UTIL__TRANSCRIPT_UPDATER_HPP_
#define WHISPER_UTIL__TRANSCRIPT_UPDATER_HPP_


#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <regex>
#include <cctype>
#include <fstream>
#include <sstream>

#include "transcript_data.hpp"

#define DEBUG_MODE 1  // Set to 0 to disable debug printouts

// TODO before submitting: 
//  Remove debug printouts or wrap them in #ifdef DEBUG_MODE
//  Ensure all functions are documented
//  Ensure all functions are tested
//   Add cutoff for transcript comparison
//   Keep log of 
//      - Removals,
//      - Comparisons
//      - Insertions 

namespace whisper {
class TranscriptUpdater {
public:
    TranscriptUpdater(int allowed_str_edit = 1, int allowed_str_min_len = 4, int allowed_substr_gaps = 10, float reduced_update_prob = 0.8)
        : allowed_str_edit(allowed_str_edit), allowed_str_min_len(allowed_str_min_len), allowed_substr_gaps(allowed_substr_gaps), reduced_update_prob(reduced_update_prob) {}

    int lcs_merge(TranscriptData& transcript,
                    const TranscriptData& update_p,
                    const int transcript_start=0);

    void find_lcs(const std::vector<std::string>& transcript_data,
                        const std::vector<std::string>& update_data,
                        std::vector<std::string>& lcs,
                        std::vector<int>& lcs_ids_transcript,
                        std::vector<int>& lcs_ids_update);

    void make_merge(TranscriptData& transcript,
                    const TranscriptData & update,
                    const std::vector<int>& lcs_ids_transcript,
                    const std::vector<int>& lcs_ids_update);

private:
    int allowed_str_edit;
    int allowed_str_min_len;
    int allowed_substr_gaps;
    float reduced_update_prob;

    void print_debug(const std::string& message, const bool new_line=true) const;
    void print_vector_debug(const std::vector<std::string>& vec, const bool new_line=true) const;
    void print_vector_debug(const std::vector<int>& vec, const bool new_line=true) const;
    void print_dp_table_debug(const std::vector<std::vector<int>>& dp) const;

    // String operations
    //    These make it easier to find matches between the transcript and the update text
    //    If a match is found, then the original word with the highest probability will be used 
    std::string matchingTransform(const std::string& str);
    std::string normalizeString(const std::string& str);
    std::string replaceDigitsWithNumberWords(const std::string& input);
    std::string replaceNumberWordsWithDigits(const std::string& input);
    std::string replacePunctuation(const std::string& input);
    // A map of cardinal numbers (word form) to their corresponding digit form.
    //      TODO:  Increase list of numbers
    std::unordered_map<std::string, std::string> number_map = {
        {"zero", "0"}, {"one", "1"}, {"two", "2"}, {"three", "3"}, {"four", "4"},
        {"five", "5"}, {"six", "6"}, {"seven", "7"}, {"eight", "8"}, {"nine", "9"},
        {"ten", "10"}, {"eleven", "11"}, {"twelve", "12"}, {"thirteen", "13"},
        {"fourteen", "14"}, {"fifteen", "15"}, {"sixteen", "16"}, {"seventeen", "17"},
        {"eighteen", "18"}, {"nineteen", "19"}, {"twenty", "20"}
    };
    // A list of Punctuation which will all be set to the same value when matching
    std::vector<std::string> punctuation_map = { ",", ".", "...", "?", "!", ":", ";"};
};
} // end of namespace whisper
#endif // WHISPER_UTIL__TRANSCRIPT_UPDATER_HPP_