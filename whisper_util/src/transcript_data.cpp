#include "transcript_data.hpp"

void TranscriptData::append(const TranscriptData& update, const size_t count, const int update_start){
    transcript.insert(transcript.end(), update.transcript.begin() + update_start, 
                                update.transcript.begin() + update_start + count);
    p.insert(p.end(), update.p.begin() + update_start, 
                                update.p.begin() + update_start + count);
    token_count.insert(token_count.end(), update.token_count.begin() + update_start, 
                                update.token_count.begin() + update_start + count);
}

float TranscriptData::get_prob(const int start, const int end) const{
    return std::accumulate(p.begin() + start, p.begin() + end, 0.0) / (end - start);
}


void TranscriptData::replace(const TranscriptData& update, const size_t count, 
                        const int update_start, const int transcript_start) {
    for (auto i=0; i<count; i++) {
        transcript[transcript_start + i] = update.transcript[update_start + i];
        p[transcript_start + i] = update.p[update_start + i];
        token_count[transcript_start + i] = update.token_count[update_start + i];
    }
}

void TranscriptData::insert(const TranscriptData& update, const size_t count, 
                        const int update_start, const int transcript_start) {
    transcript.insert(transcript.begin() + transcript_start, 
                        update.transcript.begin() + update_start, 
                        update.transcript.begin() + update_start + count);
    p.insert(p.begin() + transcript_start, 
                        update.p.begin() + update_start, 
                        update.p.begin() + update_start + count);
    token_count.insert(token_count.begin() + transcript_start, 
                        update.token_count.begin() + update_start, 
                        update.token_count.begin() + update_start + count);
}

void TranscriptData::remove(size_t index, size_t count) {
    // Remove `count` elements starting from `index`
    transcript.erase(transcript.begin() + index, transcript.begin() + index + count);
    p.erase(p.begin() + index, p.begin() + index + count);
    token_count.erase(token_count.begin() + index, token_count.begin() + index + count);
}

void TranscriptData::update_word(const int idx, const float prob) {
    p[idx] = (p[idx] * token_count[idx] + prob) / (token_count[idx] + 1);
    token_count[idx]++;
}

void TranscriptData::replace_word(const int idx, const std::string word, const float prob) {
    // TODO:  Track replaced words?
    transcript[idx] = word;
    p[idx] = prob;
    token_count[idx] = 1;
}

void TranscriptData::update_replace_word(const int idx, const TranscriptData& update, const int update_idx) {
    if (update.transcript[update_idx] == transcript[idx]) {
        // Check for exact match of token, simply increase count
        update_word(idx, update.p[update_idx]);
    } else {
        // Approximate match of words.  Take the one with higher confidence
        if (update.p[update_idx] > p[idx]) {
            replace_word(idx, update.transcript[update_idx], update.p[update_idx]);
        } else {
            // Update is less likely, reduce probability
            p[idx] /= 2.0;
        }
    }
}

std::vector<std::string> TranscriptData::splice(const int start, const int count) const {
    return std::vector<std::string>(transcript.begin() + start, transcript.begin() + start + count);
}
