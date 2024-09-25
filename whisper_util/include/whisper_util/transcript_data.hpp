#include <vector>
#include <string>
#include <numeric> // accumulate

class TranscriptData {
public:
    std::vector<std::string> transcript;
    std::vector<float> p;
    std::vector<int> token_count;

    TranscriptData(const std::vector<std::string>& transcript = {},
                   const std::vector<float>& transcript_p = {},
                   const std::vector<int>& transcript_token_count = {})
        : transcript(transcript), p(transcript_p), 
            token_count(std::vector<int>(transcript.size(), 1)) {}


    void append(const TranscriptData& update, const size_t count, const int update_start=0);
    void replace(const TranscriptData& update, const size_t count, 
                        const int update_start, const int transcript_start);
    void insert(const TranscriptData& update, const size_t count, 
                        const int update_start, const int transcript_start);
    void remove(size_t index, size_t count=1);

    void update_word(const int idx, const float prob);
    void replace_word(const int idx, const std::string word, const float prob);
    void update_replace_word(const int idx, const TranscriptData& update, const int update_idx);

    float get_prob(const int start, const int end) const;
    float get_prob(const int idx) const {
        return p[idx];
    }

    // void remove_if(const int idx);

    std::vector<std::string> splice(const int start, const int count) const;
    size_t size() const {
        return transcript.size();
    }
};
