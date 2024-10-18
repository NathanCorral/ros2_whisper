#ifndef WHISPER_UTIL__TRANSCRIPT_DATA_HPP_
#define WHISPER_UTIL__TRANSCRIPT_DATA_HPP_

#include <vector>
#include <string>
#include <numeric> // accumulate
#include <stdexcept>
#include <cstdint>
#include <algorithm> // reverse
#include <tuple>

// For tests
// #include <cassert> tests
#include <cassert>
#include <cstdlib>  // for abort()
#include <iostream>

namespace whisper {

// Classes
class _SingleToken;
class _Word;
class _Data;

// Algorithms
inline std::tuple<std::vector<int>, std::vector<int>> lcs_with_gaps(
                                                        const std::vector<std::string>& text_a,
                                                        const std::vector<std::string>& text_b,
                                                        int allowed_gaps);

// Tests
inline void my_assert(bool condition, const char* condition_str, const char* file, int line, const char* function, const char* message) {
    if (!condition) {
        std::cerr << "Assertion failed: (" << condition_str << "), function " << function
                  << ", file " << file << ", line " << line << "." << std::endl;
        if (message) {
            std::cerr << "Message: " << message << std::endl;
        }
        std::abort();  // Terminates the program
    }
}
#define MY_ASSERT(cond) my_assert((cond), #cond, __FILE__, __LINE__, __FUNCTION__, (""))
inline void test_Words(); 
inline void test_Words_0(); 
inline void test_Word();
inline void test_SingleToken();
inline void test_HelperFunctions();
inline void test_lcs();
inline void test_merge();
inline void test_combine_words();

// Helper functions
inline std::string combine(const std::vector<_SingleToken> &tokens);
inline std::string combine(const std::vector<_SingleToken> &tokens, const int start_idx, const int num_tokens);
inline std::string combine(const std::vector<std::string> &tokens, const int start_idx, const int num_tokens);
inline std::string combine(const std::vector<std::string> &tokens);
inline float combine(const std::vector<float> &probs, const int start_idx, const int num_tokens);
inline std::string combine(const std::vector<_Word> &words, const int start_idx, const int num_tokens);
inline std::string combine(const std::vector<_Word> &words);

inline int avg_occurrences(const std::vector<_SingleToken> &tokens);
inline std::vector<_Word> construct_words(const std::vector<std::string> &tokens, 
                                                    const std::vector<float> &probs);
inline void find_and_replace(std::string& str, 
          const std::string& find_str, const std::string& replace_str);
inline void replaceNumberWordsWithDigits(std::string &input);
inline std::string tolowerCase(const std::string &input);
inline void replacePunctuation(std::string& input);

inline bool special_token(const std::vector<std::string>& tokens, const int idx);
inline std::pair<bool, int> join_tokens(const std::vector<std::string>& tokens, const int idx);
inline bool my_ispunct(const std::string &token);

// Print functions
inline void print_mathching_ids(const std::tuple<std::vector<int>, std::vector<int>>& result);
inline void print_words(const std::vector<_Word> words);

class _SingleToken {
public:
  std::string text;
  float prob;
  int occurrences;

  _SingleToken(const std::string& text, float prob)
        : text(text), prob(prob), occurrences(1) {};

  _SingleToken(const std::string& text, float prob, int occ)
        : text(text), prob(prob), occurrences(occ) {};

  // Copy constructor
  _SingleToken(const _SingleToken& other)
      : text(other.text), prob(other.prob), occurrences(other.occurrences) {};

  // Move constructor
  _SingleToken(_SingleToken&& other) noexcept
      : text(std::move(other.text)), prob(other.prob), occurrences(other.occurrences) {};

  // Copy assignment operator
  _SingleToken& operator=(const _SingleToken& other) {
    if (this != &other) {
      text = other.text;
      prob = other.prob;
      occurrences = other.occurrences;
    }
    return *this;
  }

  // Move assignment operator
  _SingleToken& operator=(_SingleToken&& other) noexcept {
    if (this != &other) {
      text = std::move(other.text);
      prob = other.prob;
      occurrences = other.occurrences;
    }
    return *this;
  }

  // Comparison operators
  bool operator==(const _SingleToken& other) const {
    return text.size() == other.text.size() && text == other.text;
  }
  bool operator!=(const _SingleToken& other) const {
    return !(*this == other);
  }
  bool operator<(const _SingleToken& other) const {
    return occurrences < other.occurrences;
  }
  bool operator>(const _SingleToken& other) const {
    return other < *this;
  }
  bool operator<=(const _SingleToken& other) const {
    return !(*this > other);
  }
  bool operator>=(const _SingleToken& other) const {
    return !(*this < other);
  }
};

class _Word {
public:
  // tokens[0] -> Best word choice
  // tokens[1] -> Second best word
  // tokens[n] -> ...
  std::vector<std::vector<_SingleToken>> word_tokens;
  std::vector<int> word_occurances;
  std::vector<std::string> comparable_text;

  _Word(std::vector<_SingleToken> tokens) {
    add(tokens);
  };
  _Word(_SingleToken token) {
    add({token});
  };
  _Word() {
  };

  void inc() {
    if (!empty()) {
      word_occurances[0]++;
    }
  }

  int get_occurrences() const {
    return word_occurances[0];
  }

  void clear() {
    word_tokens.clear();
    word_occurances.clear();
    comparable_text.clear();
  }

  std::string get_comparable() const {
    return comparable_text[0];
  }

  std::string get() const {
    if (empty()) {
      return "";
    }
    return combine(word_tokens[0]);
  }

  bool empty() const {
    return word_tokens.empty() || word_tokens[0].empty();
  }

  void add(std::vector<_SingleToken> new_word) {
    word_tokens.push_back(new_word);
    word_occurances.push_back(1);
    // Only compute when swapped if needed
    comparable_text.push_back("");
    compute_comparable();
  }

  void swap(const int new_best) {
    std::swap(word_tokens[0], word_tokens[new_best]);
    std::swap(word_occurances[0], word_occurances[new_best]);
    std::swap(comparable_text[0], comparable_text[new_best]);
    compute_comparable();
  }

  void compute_comparable() {
    if (!comparable_text[0].empty()) {
      return;
    }
    std::string temp = tolowerCase(combine(word_tokens[0]));
    replaceNumberWordsWithDigits(temp);
    replacePunctuation(temp);
    comparable_text[0] = temp;
  }

  void conflict(const _Word& other) {
    if (other.empty()) {
      return;
    }

    // Check if other word exists already in conflict list
    auto best_other = other.word_tokens[0];
    // auto other_str = combine(best_other);
    for (int i=0; i<word_tokens.size(); i++) {
      // bool is_equal = other_str == combine(word_tokens[i]);
      bool is_equal = (best_other.size() == word_tokens[i].size() && 
            std::equal(best_other.begin(), best_other.end(), word_tokens[i].begin()));
      if (is_equal) {
        // Increase the count of every token in the word, swap if better than the best
        // std::for_each(word_tokens[i].begin(), word_tokens[i].end(), [](_SingleToken& token) {
        //     token.occurrences++;
        // });
        word_occurances[i]++;
        // if (i != 0 && avg_occurrences(word_tokens[i]) > avg_occurrences(word_tokens[0])) {
        if (i != 0 && word_occurances[i] > word_occurances[0]) {
          swap(i);
        }
        return;
      }
    }

    // Otherwise, add to the conflict list
    add(best_other);
    // Swap if the same as original, taking the latest version
    if (get_occurrences() == 1) {
      swap(word_tokens.size()-1);
    }
  }

  void print_all() {
    if (empty()) {
      return;
    }
    printf("Count | 'tokens1', 'tokens2', ...\n");
    for (int i=0; i<word_tokens.size(); i++) {
      printf("%d | ", avg_occurrences(word_tokens[i]));
      for (auto& token : word_tokens[i]) {
        printf("'%s', ", token.text.c_str());
      }
      printf("\n");
    }
  }
};

class _Data {
public:
  std::vector<_Word> words;
  int update_idx, prev_update_idx;
  
  _Data(): update_idx(0), prev_update_idx(0) {};


  void insert(const std::vector<_Word> &new_words, int start_idx) {
    printf("Inserting words '%s', between '%s' and '%s'\n", combine(new_words).c_str(),
                            start_idx == 0 ? "" : words[start_idx-1].get().c_str(),
                            start_idx == words.size() ? "" : words[start_idx].get().c_str());
    words.insert(words.begin()+start_idx, new_words.begin(), new_words.end());
  }
  void insert(const _Word &new_word, int start_idx) {
    printf("Insert word '%s', between '%s' and '%s'\n", new_word.get().c_str(),
                            start_idx == 0 ? "" : words[start_idx-1].get().c_str(),
                            start_idx == words.size() ? "" : words[start_idx].get().c_str());
    words.insert(words.begin()+start_idx, new_word);
  }

  void print() const {
    for (const auto &word : words) {
      printf("'%s', ", word.get().c_str());
    }
  }

  void print(int n) const {
    // Print tokens with > n occurrences
    for (const auto &word : words) {
      if (word.get_occurrences() < n) {
        continue;
      }
      printf("'%s', ", word.get().c_str());
    }
  }

  void merge(const std::vector<_Word> &new_words) {
    // TODO:  Set  update_idx
    // TODO:  Modify lcs algo to handled allowed gaps between elements
    // TODO:  Decrement counts words that are between matching substrings and are not present in the udpate.
    // TODO:  Test if all punctuation should be handled the same.
    const int allowed_gaps = 5;
    // First case
    if(words.empty()) {
      printf("Adding first words!\n");
      insert(new_words, words.size());
      return;
    }
    std::string cur_words_str = combine(words, update_idx, words.size()-update_idx);
    std::string new_words_str = combine(new_words, 0, new_words.size());
    printf("Current words: '%s'\n", cur_words_str.c_str());
    printf("       Update: '%s'\n", new_words_str.c_str());

    std::vector<std::string> compA;
    std::vector<std::string> compB;

    for (int i=update_idx; i<words.size(); i++) {
      compA.push_back(words[i].get_comparable());
    }
    for (int j=0; j<new_words.size(); j++) {
      compB.push_back(new_words[j].get_comparable());
    }

    auto cur_words_comp = combine(compA);
    auto new_words_comp = combine(compB);
    printf(" --- comparable: '%s'\n", cur_words_comp.c_str());
    printf("       new comp: '%s'\n", new_words_comp.c_str());

    auto [indiciesA, indiciesB] = lcs_with_gaps(compA, compB, allowed_gaps);
    if (indiciesA.empty()) {
      printf("No overlap \n");  
      insert(new_words, words.size());
      return;
    }

    // (temp) Print results
    printf("Matching words:  \n");
    for (int i=0; i<indiciesA.size(); i++) {
      printf("(%d, %d): '%s'\n", indiciesA[i], indiciesB[i], 
                    words[indiciesA[i]+update_idx].get().c_str());
    }


    // 4. Increase word count of current match
    int prevA = indiciesA[0];    
    int prevB = indiciesB[0];
    words[prevA + update_idx].inc();

    // Keep track of the insertions/deletions to perform.
    //   so they can simply be applied at the end without worrying about offsets
    std::vector<std::pair<int, int>> pending_inserts;
    std::vector<int> pending_deletions;


    // For every other matching word
    for(int i=1; i<indiciesA.size(); i++) {
      int curA = indiciesA[i];
      int curB = indiciesB[i];
      int gapA = curA - prevA;
      int gapB = curB - prevB;

      // Increase the count for the matching token
      words[indiciesA[i] + update_idx].inc();

      if (gapA == 1 && gapB == 1) {
        // Continuous sub string
        prevA = curA;
        prevB = curB;
      } 
      else if (gapA == 1) {
        // There are words in the original which appear in the update, insert them
        for (int j=prevB; j<curB; j++) {
          pending_inserts.push_back({prevA+update_idx, j});
        }
      }
      else if (gapB == 1) {
        // The original has words missing in the update
        // TODO:  Decrement their count!
      }
      else {
        // Conflict, different words in original and update
        prevA++;
        prevB++;
        while (prevA < curA && prevB < curB) {
          printf("Conflict between %s and %s\n", words[prevA + update_idx].get().c_str(), 
                                                          new_words[prevB].get().c_str());
          words[prevA + update_idx].conflict(new_words[prevB]);
          prevA++;

          prevB++;
        }
        // Insert remaining new words
        while (prevB < curB) {
          pending_inserts.push_back({prevA+update_idx, prevB});
          prevB++;
        }
      }
      prevA = curA;
      prevB = curB;
    }
    // Increment indices for any remaining words
    // Perform last conflicts
    prevA++;
    prevB++;
    while (prevA < words.size() && prevB < new_words.size()) {
      printf("Final Conflict between %s and %s\n", words[prevA + update_idx].get().c_str(), 
                                                      new_words[prevB].get().c_str());
      words[prevA + update_idx].conflict(new_words[prevB]);
      prevA++;
      prevB++;
    }

    // Perform last inserts
    while(prevB < new_words.size()) {
      pending_inserts.push_back({words.size(), prevB});
      prevB++;
    }


    // Perform the insertions
    int offset = 0;
    for (auto& [idx, new_idx] : pending_inserts) {
      insert(new_words[new_idx], idx + offset);
      offset++;
    }
    // Perform removals
    // TODO 
  }

};



// Helper functions
// inline std::string combine(std::vector<_SingleToken> tokens) {
//   return std::accumulate(tokens.begin(), tokens.end(), std::string(""),
//         [](const std::string& acc, const _SingleToken& token) {
//             return acc + token.text;
//           });
// }

inline std::string combine(const std::vector<_SingleToken> &tokens) {
  return combine(tokens, 0, tokens.size());
}
inline std::string combine(const std::vector<_SingleToken> &tokens, 
                              const int start_idx, const int num_tokens) {
  return std::accumulate(tokens.begin() + start_idx, 
                          tokens.begin() + start_idx + num_tokens, std::string(""),
                          [](const std::string& acc, const _SingleToken& token) {
                              return acc + token.text;
  });
}
inline std::string combine(const std::vector<std::string> &tokens, 
                          const int start_idx, const int num_tokens) {
  if (start_idx < 0 || start_idx >= tokens.size()) {
    throw std::out_of_range("start_idx is out of range");
  }
  if (num_tokens < 0 || start_idx + num_tokens > tokens.size()) {
    throw std::out_of_range("num_tokens goes out of range");
  }
  return std::accumulate(tokens.begin() + start_idx, 
                          tokens.begin() + start_idx + num_tokens, std::string(""));
}
inline std::string combine(const std::vector<std::string> &tokens) {
  return combine(tokens, 0, tokens.size());
}
inline float combine(const std::vector<float> &probs, 
                          const int start_idx, const int num_tokens) {
  return std::accumulate(probs.begin() + start_idx, 
                          probs.begin() + start_idx + num_tokens, 0);
}
inline std::string combine(const std::vector<_Word> &words, 
                          const int start_idx, const int num_tokens) {
  return std::accumulate(words.begin() + start_idx, 
                          words.begin() + start_idx + num_tokens, std::string(""),
                          [](const std::string& acc, const _Word& word) {
                              return acc + word.get();
  });
}
inline std::string combine(const std::vector<_Word> &words) {
  return combine(words, 0, words.size());
}

inline int avg_occurrences(const std::vector<_SingleToken> &tokens) {
  if (tokens.empty()) {
    return 0;
  }
  // Since all tokens in the word should have the same number of occurrences
  // return static_cast<float>(tokens[0].occurrences);
  return tokens[0].occurrences;
}

inline void find_and_replace(std::string& str, 
          const std::string& find_str, const std::string& replace_str) {
  size_t pos = str.find(find_str);
  if (pos != std::string::npos) {
    str.replace(pos, find_str.size(), replace_str);
  }
}

inline std::string tolowerCase(const std::string &input) {
  std::string result;
  for (char ch : input) {
    // result += std::tolower(static_cast<unsigned char>(ch));
    result += std::tolower(ch);
  }
  return result;
}

inline void replaceNumberWordsWithDigits(std::string &input) {
  const std::unordered_map<std::string, std::string> number_map = {
      {"zero", "0"}, {"one", "1"}, {"two", "2"}, {"three", "3"}, {"four", "4"},
      {"five", "5"}, {"six", "6"}, {"seven", "7"}, {"eight", "8"}, {"nine", "9"},
      {"ten", "10"}, {"eleven", "11"}, {"twelve", "12"}, {"thirteen", "13"},
      {"fourteen", "14"}, {"fifteen", "15"}, {"sixteen", "16"}, {"seventeen", "17"},
      {"eighteen", "18"}, {"nineteen", "19"}, {"twenty", "20"}
  };
  for (const auto& pair : number_map) {
    find_and_replace(input, pair.first, pair.second);
  }
}


inline void replacePunctuation(std::string& input) {
  const std::vector<std::string> punctuation_map = { ",", ".", "...", "?", "!", ":", ";"};
  for (const auto& punc : punctuation_map) {
    find_and_replace(input, punc, "_");
  }
}



inline bool special_token(const std::vector<std::string>& tokens, const int idx) {
  // Determine if we should skip the current token (if is is a whisper-specific directive)
  //   Return:  bool -- Skip the current token
  const std::vector<std::string> skip_starts = {
      "[_BEG_]", "[_TT_", " [_BEG_]", " [_TT_"
  };

  // Boundary check
  if (tokens.size() <= idx) {
    return false;
  }

  // See if token starts with special indicators
  for (auto &start : skip_starts) { 
    if (tokens[idx].size() >= start.size() && tokens[idx].substr(0, start.size()) == start) {
      return true;
    }
  }
  return false;
}

inline std::pair<bool, int> join_tokens(const std::vector<std::string>& tokens, const int idx) {
  // Check if, starting from the idx, the tokens are a bracket which can be combined or removed
  //   Return:
  //      bool -- Tokens start with bracket (can be combined)
  //      int  -- Number of tokens to combine
  const std::vector<std::pair<std::string, std::string>> combine_brackets = {
      {"[", "]"}, {" [", "]"},
      {"{", "}"}, {" {", "}"},
      {"(", ")"}, {" (", ")"}
  };
  // Set a limit on how many tokens can be within brackets
  const size_t max_allowed_tokens_to_combine = 10;

  // Boundary check
  if (tokens.size() <= idx) {
    return {false, 0};
  }

  // Try to combine tokens
  for (auto &[start, end] : combine_brackets) { 
    if (tokens[idx].size() >= start.size() && tokens[idx].substr(0, start.size()) == start) {
      bool end_found = false;
      int end_idx = idx + 1;
      while (end_idx < tokens.size() && (end_idx-idx) <= max_allowed_tokens_to_combine) {
        if (tokens[end_idx].size() >= end.size() && tokens[end_idx].substr(0, end.size()) == end) {
          return {true, end_idx - idx + 1};
        }
        end_idx++;
      }
      // We found the start but not the end, "combine" current token with itself
      return {true, 1};
    }
  }
  return {false, 0};
}

inline bool my_ispunct(const std::string &token) {
  // The reason for not using std::punct is that "'t" would be considered a punctuation.
  // As well as some brackets which should be combined into words.
  // Otherwise it is missing "..." which we can consider punctuation
  const std::vector<std::string> punctuations = {",", ".", "?", "!", ":", ";", "...", "+", "-"};
  return std::find(punctuations.begin(), punctuations.end(), token) != punctuations.end();
}


inline std::pair<bool, int> preprocess(const std::vector<std::string>& texts, int idx) {
  // Check if, starting from the idx, the tokens are a bracket which can be combined or removed
  //   Return:
  //      bool -- Tokens start with bracket (can be combined or skipped)
  //      int  -- Number of tokens (negative indicates skip, positive indicates combine)
  const std::vector<std::pair<std::string, std::string>> combine_brackets = {
      {"[", "]"}, {" [", "]"},
      {"{", "}"}, {" {", "}"},
      {"(", ")"}, {" (", ")"}
  };
  const std::vector<std::pair<std::string, std::string>> skip_brackets = {
      {"[_", "]"}
  };
  // Set a limit on how many tokens can be within brackets
  const size_t max_allowed_tokens_to_combine = 10;

  // Boundary check
  if (texts.size() <= idx) {
    return {false, 0};
  }

  // Try to combine tokens
  for (auto &[start, end] : combine_brackets) { 
    if (texts[idx].size() >= start.size() && texts[idx].substr(0, start.size()) == start) {
      bool end_found = false;
      int end_idx = idx+1;
      while (end_idx < texts.size() && (end_idx-idx) <= max_allowed_tokens_to_combine) {
        if (texts[end_idx].size() < end.size() && texts[end_idx].substr(0, end.size()) == end) {
          end_found = true;
          break;
        }
        // Return since we found the start token, indicate success based on the end being found
        return {end_found, end_idx-idx};
      }
    }
  }

  // Check for special (whisper) tokens to skip
  for (auto &[start, end] : skip_brackets) { 
    if (texts[idx].size() < start.size() && texts[idx].substr(0, start.size()) == start) {
      // Start begins with "[_", just skip the token (which will end with "]")
      return {true, -1};
    }
  }

  return {false, 0};
}

inline std::vector<_Word> construct_words(const std::vector<std::string> &tokens, 
                                                    const std::vector<float> &probs) {
  std::vector<_Word> words;
  std::vector<_SingleToken> word_wip;
  for (int i=0; i<tokens.size(); i++) {
    // printf("(%d) Current token:  '%s'\n", i, tokens[i].c_str());

    // Decide if we should start a new word
    if(!word_wip.empty() && !tokens[i].empty()) {
      if(std::isspace(tokens[i][0])) {
        // If the token starts with a space, it is start of new word
        words.push_back({word_wip});
        word_wip.clear();
      }
      else if (my_ispunct(tokens[i])) {
        // Add last word
        words.push_back({word_wip});
        word_wip.clear();
        // Add punctuation as its own word
        words.push_back(_SingleToken(tokens[i], probs[i])); 
        // Current token is already used in a word so no need to continue rest of loop
        continue;
      }
    }

    if (special_token(tokens, i)) {
      // Do not add anything to word_wip
      // printf("\tSkipping special token..\n");
    }
    else if (auto [join, num_tokens] = join_tokens(tokens, i); join) {

      std::string combined_text = combine(tokens, i, num_tokens);
      float prob = combine(probs, i, num_tokens);
      
      // printf("\tCombining next %d tokens-> '%s'", num_tokens, combined_text.c_str());
      
      word_wip.push_back(_SingleToken(combined_text, prob));
      i += num_tokens - 1;
    }
    else {
      word_wip.push_back(_SingleToken(tokens[i], probs[i]));
    }

    // Decide if we should start a new word
    // if( (i+1) < tokens.size() && !tokens[i+1].empty() && std::isspace(tokens[i+1][0])) {
    //   if (word_wip.size() > 0) {
    //     printf("\tNew word created!\n");
    //     words.push_back({word_wip});
    //     word_wip.clear();
    //   }
    // }

    /*
    // Check if we can combine the current token into a whisper-specific directive
    auto [combine_or_skip, num_tokens] = preprocess(tokens, i);
    if (combine_or_skip) {
      if (num_tokens < 0) {
        i += std::abs(num_tokens) - 1;
        continue;
      }

      // Combine the tokens
      std::string combined_text = combine(tokens, i, num_tokens);
      float prob = combine(probs, i, num_tokens);
      word_wip.push_back(_SingleToken(combined_text, prob));
    } else {
      // Add token to the current word
      word_wip.push_back(_SingleToken(tokens[i], probs[i]));
    }
    */

    /*
    // if next token begins with space, create new word
    if( (i+1) < tokens.size() && !tokens[i+1].empty() && tokens[i+1].substr(0, 1) == " ") {
      words.push_back({word_wip});
      word_wip.clear();
    }
    */
  }

  // Final word
  if (!word_wip.empty()) {
    words.push_back({word_wip});
  }

  return words;
}

// Print functions
inline void print_mathching_ids(const std::tuple<std::vector<int>, std::vector<int>>& result) {
  const auto& [indices_a, indices_b] = result;
  std::cout << "Indices in text_a: ";
  for (int i : indices_a) std::cout << i << " ";
  std::cout << "\nIndices in text_b: ";
  for (int i : indices_b) std::cout << i << " ";
  std::cout << "\n";
}
inline void print_words(const std::vector<_Word> words) {
  // Use '', to separate each word
  // Use | to separate each token
  for (auto& word : words) {
    printf("'");
    for (auto& token : word.word_tokens[0]) {
      printf("%s|", token.text.c_str());
    }
    printf("', ");
  }
  printf("\n");
}


// Tests
inline void test_Words() {
  test_Words_0();
  test_SingleToken();
  test_HelperFunctions();
  test_Word();
  test_lcs();
  test_combine_words();
  // test_merge();

  return;
}

inline void test_Words_0() {
  _SingleToken token1{"Hel", 0.9f};
  _SingleToken token2{"lo", 0.85f};
  
  _Word word;
  word.add({token1, token2});
  
  std::string best_word = word.get();
  printf("Best Word: '%s'\n", best_word.c_str());
  return;
}

// Test function for _SingleToken
inline void test_SingleToken() {
    // Test constructor and initialization
    _SingleToken token1{"test", 0.75f};
    MY_ASSERT(token1.text == "test");
    MY_ASSERT(token1.prob == 0.75f);
    MY_ASSERT(token1.occurrences == 1);

    // Test copy constructor
    _SingleToken token2 = token1;
    MY_ASSERT(token2.text == "test");
    MY_ASSERT(token2.prob == 0.75f);
    MY_ASSERT(token2.occurrences == 1);
    
    // Test move constructor
    _SingleToken token3 = std::move(token2);
    MY_ASSERT(token3.text == "test");
    MY_ASSERT(token3.prob == 0.75f);
    MY_ASSERT(token3.occurrences == 1);

    // Test copy assignment operator
    _SingleToken token4{"new", 0.5f};
    token4 = token1;
    MY_ASSERT(token4.text == "test");
    MY_ASSERT(token4.prob == 0.75f);
    MY_ASSERT(token4.occurrences == 1);

    // Test move assignment operator
    _SingleToken token5{"move", 0.1f};
    token5 = std::move(token4);
    MY_ASSERT(token5.text == "test");
    MY_ASSERT(token5.prob == 0.75f);
    MY_ASSERT(token5.occurrences == 1);

    // Test comparison operators
    MY_ASSERT(token1 == token5);
    MY_ASSERT((token1 != _SingleToken{"diff", 0.9f}));
    MY_ASSERT((token1 < _SingleToken{"other", 0.9f, 2}));  // Occurrences comparison
    MY_ASSERT(token1 <= token5);
    MY_ASSERT((_SingleToken{"other", 0.9f, 2} > token1));

    printf("test_SingleToken passed.\n");
}

// Test function for combine and avg_occurrences
inline void test_HelperFunctions() {
    _SingleToken token1{"A", 0.9f};
    _SingleToken token2{"B", 0.85f};
    std::vector<_SingleToken> tokens = {token1, token2};
    
    // Test combine function
    std::string combined = combine(tokens);
    MY_ASSERT(combined == "AB");

    // Test avg_occurrences function
    MY_ASSERT(avg_occurrences(tokens) == 1);  // All occurrences are the same

    // Test empty case for avg_occurrences
    std::vector<_SingleToken> empty_tokens;
    MY_ASSERT(avg_occurrences(empty_tokens) == 0);

    printf("test_HelperFunctions passed.\n");
}

// Test function for _Word class
inline void test_Word() {
    _SingleToken token1{"Hel", 0.9f};
    _SingleToken token2{"lo", 0.85f};

    _Word word;
    word.add({token1, token2});

    // Test get() and empty()
    MY_ASSERT(word.get() == "Hello");
    MY_ASSERT(!word.empty());

    // Test empty word case
    _Word empty_word;
    MY_ASSERT(empty_word.get() == "");
    MY_ASSERT(empty_word.empty());

    // Test conflict resolution
    _SingleToken token3{"He", 0.8f};
    _SingleToken token4{"llo", 0.9f};

    _Word word2;
    word2.add({token3, token4});
    word.print_all();
    word.conflict(word2);
    word.print_all();
    word.conflict(word); // Increment count
    word.print_all();

    // Verify if conflict resolution updated occurrences    
    MY_ASSERT(word.word_tokens[0][0].occurrences == 1);  // "Hel" and "He" should merge
    MY_ASSERT(word.word_tokens[0][1].occurrences == 1);  // "lo" and "llo" should merge

    printf("test_Word passed.\n");
}


inline void test_lcs() {
  std::vector<std::string> text_a = {"apple", "banana", "cherry", "date", "grape"};
  std::vector<std::string> text_b = {"apple", "blueberry", "cherry", "date", "elderberry", "grape"};
  int allowed_gaps = 4;

  auto result = lcs_with_gaps(text_a, text_b, allowed_gaps);
  print_mathching_ids(result);

  return;
}



inline void test_combine_words() {
  std::vector<std::vector<std::string>> updates = {
        {" [", "BL", "ANK", "_", "AUD", "IO", "]"},
        {" Chapter", " 7", "."},
        {" [_BEG_]", " I", " don", "'t", " remember", "[_TT_410]", "[_BEG_]", " [", "BL", "ANK", "_", "AUD", "IO", "]", "[_TT_500]" },
        // {" [", "BL", "ANK", "_", "AUD", "IO", "]"},
        // {" [", "BL", "ANK", "_", "AUD", "IO", "]"},
        // {" [", "BL", "ANK", "_", "AUD", "IO", "]"},
        // {" Chapter", " 7", "."},
        // {" Chapter", ",", " chapter", " six", "."},
        // {" Chapter", ",", " chapter", " six", ",", " the", " journey", " for", " the", " Lord", "."},
        // {" Chapter", ",", " chapter", " six", ",", " the", " journey", " of", " journey", " from", " platform", "."},
        // {" Chapter", ",", " chapter", " six", ".", " The", " journey", " from", " platform", " nine", " and", " four", "."},
        // {" Chapter", ",", " chapter", " 6", ",", " the", " journey", " from", " platform", " 9", " and", " 3", " chord", "."},
        // {" to", " six", ",", " the", " journey", " from", " platform", " nine", " and", " three", " chord", " chords", "."},
        // {" The", " journey", " from", " platform", " 9", " and", " 3", " chord", " chords", ".", " [", "BL", "ANK", "_", "AUD", "IO", "]"},
        // {" only", " from", " platform", " nine", " and", " nine", " and", " three", " chord", " chords", ".", " [", "BL", "ANK", "_", "AUD", "IO", "]"},
        // {" Form", " nine", " and", " nine", " and", " three", " quarters", ".", " (", "cl", "aps", ")"},
        // {" 9", " and", " 3", " quarters", ".", " (", "cl", "aps", ")", " -", " Harris", " last", " month", " with", " the", "..."},
        // {" [", "cl", "aps", "]", " Harry", "'s", " last", " month", " with", " the", " t", "urd", "s", " list", " was", "..."},
        // {" Harry", "'s", " last", " month", " with", " the", " D", "urs", "leys", " wasn", "'t", " fun", "."}, // <TODO: Check this line again!!
        // {" Harry", "'s", " last", " month", " with", " the", " t", "urd", "s", " list", " wasn", "'t", " fine", "."},
        // {" The", " first", " month", " with", " the", " d", "urs", "leys", " wasn", "'t", " fine", ".", " True", ",", " W", " was", " now", " so", "..."},
        // {" with", " the", " d", "urs", "leys", " wasn", "'t", " fun", ".", " True", " W", " was", " now", " so", " scared", "."},
        // {" Was", "n", "'t", " fine", ".", " True", ",", " W", " was", " now", " so", " scared", " of", " Harry", "."},
        // {" True", ",", " Doug", ",", " W", " was", " now", " so", " scared", " of", " Harry", "."},
        // {" True", ",", " Doug", ",", " Doug", " Lee", " was", " now", " so", " scared", " of", " Harry", ".", " Harry", ",", " he", " wouldn", "'t", " stay", " in", " the", " same", " room", "."},
        // {" W", " was", " now", " so", " scared", " scared", " of", " Harry", " Harry", " he", " wouldn", "'t", " stay", " in", " the", " same", " room", " in", " room"},
        // {" I", "'m", " so", " scared", " scared", " of", " Harry", " Harry", " he", " wouldn", "'t", " stay", " in", " the", " same", " room", " in", " room", " while", " on"},
        // {" scared", " of", " Harry", " Harry", " he", " wouldn", "'t", " stay", " and", " stay", " in", " the", " same", " room", " in", " room", " while", " aren", "'t", " the", " ch", "ump", " of", " junior", " and", " I", "'m"},
        // {" he", " wouldn", "'t", " stay", " in", " the", " same", " room", " in", " room", " while", " I", "'m", " pet", "unia", " and", " I", "'m", " here", " and", " uncle", " Vernon"},
        // {" stay", " in", " the", " same", " room", " in", " room", " while", " I", "'m", " pet", "unia", " and", " uncle", " Vernon", " Vernon", " didn", "'t", " shut", " him"},
        // {" room", " while", " I", "'m", " pet", "unia", " and", " uncle", " Vernon", " Vernon", " didn", "'t", " shut", " him", " shut", " having", " it"},
        // {" While", " Aunt", " Pat", "un", ",", " here", " and", " on", " here", ",", " and", " Uncle", " Vernon", ",", " Vernon", " didn", "'t", " shut", " his", " shut", " having", " in", " his", " clean", " his", " cup", "board", "."},
        // {" P", "ach", "un", "ier", ",", " and", " on", " here", ",", " and", " Uncle", " Vernon", ",", " Vernon", " didn", "'t", " shut", " him", " shut", " Harry", " in", " his", " clean", " his", " cup", "board", ",", " forcing", " him", " to", " do"},
        // {" And", " Uncle", " Vernon", ",", " Vernon", " didn", "'t", " shut", " his", " shut", " Harry", " in", " his", " cup", "board", ",", " forcing", " him", " to", " do", " anything", "."},
        // {" and", " didn", "'t", " shut", " Harry", " in", " his", " cup", "board", ",", " force", " him", " to", " do", " anything", ",", " anything", ",", " or", " shut"}

  };

  for(auto & data : updates) {
    std::vector<float> probs = std::vector<float>(data.size(), 1.0f);
    std::vector<_Word> words = construct_words(data, probs);
    print_words(words);
  }
}

inline void test_merge() {
  std::vector<std::vector<std::string>> updates = {
        {" [", "BL", "ANK", "_", "AUD", "IO", "]"},
        {" [", "BL", "ANK", "_", "AUD", "IO", "]"},
        {" [", "BL", "ANK", "_", "AUD", "IO", "]"},
        {" Chapter", " 7", "."},
        {" Chapter", ",", " chapter", " six", "."},
        {" Chapter", ",", " chapter", " six", ",", " the", " journey", " for", " the", " Lord", "."},
        {" Chapter", ",", " chapter", " six", ",", " the", " journey", " of", " journey", " from", " platform", "."},
        {" Chapter", ",", " chapter", " six", ".", " The", " journey", " from", " platform", " nine", " and", " four", "."},
        {" Chapter", ",", " chapter", " 6", ",", " the", " journey", " from", " platform", " 9", " and", " 3", " chord", "."},
        {" to", " six", ",", " the", " journey", " from", " platform", " nine", " and", " three", " chord", " chords", "."},
        {" The", " journey", " from", " platform", " 9", " and", " 3", " chord", " chords", ".", " [", "BL", "ANK", "_", "AUD", "IO", "]"},
        {" only", " from", " platform", " nine", " and", " nine", " and", " three", " chord", " chords", ".", " [", "BL", "ANK", "_", "AUD", "IO", "]"},
        {" Form", " nine", " and", " nine", " and", " three", " quarters", ".", " (", "cl", "aps", ")"},
        {" 9", " and", " 3", " quarters", ".", " (", "cl", "aps", ")", " -", " Harris", " last", " month", " with", " the", "..."},
        {" [", "cl", "aps", "]", " Harry", "'s", " last", " month", " with", " the", " t", "urd", "s", " list", " was", "..."},
        {" Harry", "'s", " last", " month", " with", " the", " D", "urs", "leys", " wasn", "'t", " fun", "."}, // <TODO: Check this line again!!
        {" Harry", "'s", " last", " month", " with", " the", " t", "urd", "s", " list", " wasn", "'t", " fine", "."},
        {" The", " first", " month", " with", " the", " d", "urs", "leys", " wasn", "'t", " fine", ".", " True", ",", " W", " was", " now", " so", "..."},
        {" with", " the", " d", "urs", "leys", " wasn", "'t", " fun", ".", " True", " W", " was", " now", " so", " scared", "."},
        {" Was", "n", "'t", " fine", ".", " True", ",", " W", " was", " now", " so", " scared", " of", " Harry", "."},
        {" True", ",", " Doug", ",", " W", " was", " now", " so", " scared", " of", " Harry", "."},
        {" True", ",", " Doug", ",", " Doug", " Lee", " was", " now", " so", " scared", " of", " Harry", ".", " Harry", ",", " he", " wouldn", "'t", " stay", " in", " the", " same", " room", "."},
        {" W", " was", " now", " so", " scared", " scared", " of", " Harry", " Harry", " he", " wouldn", "'t", " stay", " in", " the", " same", " room", " in", " room"},
        {" I", "'m", " so", " scared", " scared", " of", " Harry", " Harry", " he", " wouldn", "'t", " stay", " in", " the", " same", " room", " in", " room", " while", " on"},
        {" scared", " of", " Harry", " Harry", " he", " wouldn", "'t", " stay", " and", " stay", " in", " the", " same", " room", " in", " room", " while", " aren", "'t", " the", " ch", "ump", " of", " junior", " and", " I", "'m"},
        {" he", " wouldn", "'t", " stay", " in", " the", " same", " room", " in", " room", " while", " I", "'m", " pet", "unia", " and", " I", "'m", " here", " and", " uncle", " Vernon"},
        {" stay", " in", " the", " same", " room", " in", " room", " while", " I", "'m", " pet", "unia", " and", " uncle", " Vernon", " Vernon", " didn", "'t", " shut", " him"},
        {" room", " while", " I", "'m", " pet", "unia", " and", " uncle", " Vernon", " Vernon", " didn", "'t", " shut", " him", " shut", " having", " it"},
        {" While", " Aunt", " Pat", "un", ",", " here", " and", " on", " here", ",", " and", " Uncle", " Vernon", ",", " Vernon", " didn", "'t", " shut", " his", " shut", " having", " in", " his", " clean", " his", " cup", "board", "."},
        {" P", "ach", "un", "ier", ",", " and", " on", " here", ",", " and", " Uncle", " Vernon", ",", " Vernon", " didn", "'t", " shut", " him", " shut", " Harry", " in", " his", " clean", " his", " cup", "board", ",", " forcing", " him", " to", " do"},
        {" And", " Uncle", " Vernon", ",", " Vernon", " didn", "'t", " shut", " his", " shut", " Harry", " in", " his", " cup", "board", ",", " forcing", " him", " to", " do", " anything", "."},
        {" and", " didn", "'t", " shut", " Harry", " in", " his", " cup", "board", ",", " force", " him", " to", " do", " anything", ",", " anything", ",", " or", " shut"}
  };
  _Data d;
  for(auto & data : updates) {
    std::vector<float> probs = std::vector<float>(data.size(), 1.0f);
    std::vector<_Word> words = construct_words(data, probs);
    if(words.size() > 5) {
      // Window: Remove last and first word
      // for( int i=0; i<1; i++) {
      //   words.erase(words.begin() + words.size());
      // }
      // for( int i=0; i<1; i++) {
      //   words.erase(words.begin());
      // }
    }
    d.merge(words);
    printf("\n");
  }
  printf("\n");
  printf("Data:  \n");
  d.print(2);
  printf("\n");
  printf("\n");
}




// Algorithms
inline std::tuple<std::vector<int>, std::vector<int>> lcs_with_gaps(
                                                        const std::vector<std::string>& text_a,
                                                        const std::vector<std::string>& text_b,
                                                        int allowed_gaps) {
    int n = text_a.size();
    int m = text_b.size();
    
    // DP table of (n+1)x(m+1)x(allowed_gaps+1)
    std::vector<std::vector<std::vector<int>>> dp(n + 1, std::vector<std::vector<int>>(m + 1, std::vector<int>(allowed_gaps + 1, 0)));
    std::vector<std::vector<std::vector<std::pair<int, int>>>> previous(n + 1, std::vector<std::vector<std::pair<int, int>>>(m + 1, std::vector<std::pair<int, int>>(allowed_gaps + 1, {-1, -1})));
    
    int max_length = 0;
    int end_i = 0, end_j = 0, end_g = 0;

    // Fill the DP table
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            for (int g = 0; g <= allowed_gaps; ++g) {
                // Case 1: Match found (text_a[i-1] == text_b[j-1])
                if (text_a[i - 1] == text_b[j - 1]) {
                    dp[i][j][g] = dp[i - 1][j - 1][g] + 1;
                    previous[i][j][g] = {i - 1, j - 1};
                }

                // Case 2: No match, consider using a gap
                if (g > 0) {
                    // Skip an element in text_a
                    if (dp[i - 1][j][g - 1] > dp[i][j][g]) {
                        dp[i][j][g] = dp[i - 1][j][g - 1];
                        previous[i][j][g] = {i - 1, j};
                    }
                    // Skip an element in text_b
                    if (dp[i][j - 1][g - 1] > dp[i][j][g]) {
                        dp[i][j][g] = dp[i][j - 1][g - 1];
                        previous[i][j][g] = {i, j - 1};
                    }
                }

                // Keep track of the maximum length and its location
                if (dp[i][j][g] > max_length) {
                    max_length = dp[i][j][g];
                    end_i = i;
                    end_j = j;
                    end_g = g;
                }
            }
        }
    }

    // Backtrack to find the indices
    std::vector<int> indices_a, indices_b;
    int i = end_i, j = end_j, g = end_g;

    while (i > 0 && j > 0) {
        auto [prev_i, prev_j] = previous[i][j][g];
        if (prev_i == i - 1 && prev_j == j - 1) {
            // This was a match
            indices_a.push_back(i - 1);
            indices_b.push_back(j - 1);
        } else if (g > 0) {
            // A gap was used, decrement the gap counter
            --g;
        }
        i = prev_i;
        j = prev_j;
    }

    // Reverse to get correct order
    reverse(indices_a.begin(), indices_a.end());
    reverse(indices_b.begin(), indices_b.end());

    return {indices_a, indices_b};
}








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
  // one-to-many mapping tokens to alternatives
  std::vector<std::vector<int>> token_clonflict_mapping; 
  std::vector<std::string> token_conflicts;     // tokens over-written
  std::vector<float> token_confilict_probs;
  std::vector<int> token_confilict_counts;

  std::vector<int> get_token_ids_right(const int text_id) const {
    return get_token_ids_within(text_id, text.size());
  }

  std::vector<int> get_token_ids_within(const int text_id_start, const int len) const {
    const int end = text_id_start + len;
    std::vector<int> ret;
    int prev_token_id = -1;
    int text_id = text_id_start;
    while (text_id < text.size() && text_id < end) {
      if (prev_token_id < 0 || prev_token_id != text_token_mapping[text_id]) {
        prev_token_id = text_token_mapping[text_id];
        ret.push_back(prev_token_id);
      }
      text_id++;
    }
    return ret;
  }

  void add_and_swap_conflict(const int text_idx, const WhisperSegment other, const int other_text_idx) {
    int token_idx = text_token_mapping[text_idx];
    int other_token_idx = other.text_token_mapping[text_idx];
    std::string other_token = other.tokens[other_token_idx];

    // Check if the token already exists in the conflicts
    int conflict_idx = -1;
    for (auto check_conflict_idx : token_clonflict_mapping[token_idx]) {
      if (other_token == token_conflicts[check_conflict_idx]) {
        conflict_idx = check_conflict_idx;
        token_confilict_counts[conflict_idx]++;
        break;
      }
    }

    // If not, add the other token to the conflicts
    if (conflict_idx < 0) {
      int new_idx = token_conflicts.size();
      token_conflicts.push_back(other_token);
      token_confilict_probs.push_back(other.probs[other_token_idx]);
      token_confilict_counts.push_back(other.counts[other_token_idx]);

      token_clonflict_mapping[token_idx].push_back(new_idx);
      conflict_idx = new_idx;
    }

    // Swap the token with the conflict
    std::swap(tokens[token_idx], token_conflicts[conflict_idx]);
    std::swap(probs[token_idx], token_confilict_probs[conflict_idx]);
    std::swap(counts[token_idx], token_confilict_counts[conflict_idx]);
  }

  WhisperSegment(const std::string & text, int64_t t0, int64_t t1)
        : text(text), t0(t0), t1(t1), audio_offset_ms(0), occurances(1) {};

  WhisperSegment(const std::string & text, int64_t t0, int64_t t1, 
                      std::vector<std::string> tokens, std::vector<float> probs)
        : text(text), t0(t0), t1(t1), audio_offset_ms(0), occurances(1),
          tokens(tokens), probs(probs)
         {
            counts = std::vector<int>(tokens.size(), 1);  
            text_token_mapping = mapCharToToken(text, tokens);
            // int max = *std::max_element(text_token_mapping.begin(), text_token_mapping.end());
            token_clonflict_mapping = std::vector<std::vector<int>>(tokens.size(), std::vector<int>());
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

  void merge_clobber_right(WhisperSegment & cur, WhisperSegment & update, 
                int cur_start, int update_start, int lcs_len) {
    std::vector<int> token_ids_in_lcs = cur.get_token_ids_within(cur_start, lcs_len);
    // std::vector<int> token_ids_right = cur.get_token_ids_right(cur_start + lcs_len);
    // std::vector<int> new_token_ids_right = update.get_token_ids_right(update_start + lcs_len);
    printf("Cur: '%s'\n", cur.text.c_str());
    printf("Update: '%s'\n", update.text.c_str());

    // Increase count of tokens within
    for (auto & id : token_ids_in_lcs) {
      cur.counts[id]++;
    }

    // Backtrack to start of current token
    int remove_text_start = cur_start + lcs_len;
    while (remove_text_start > 0 && 
              cur.text_token_mapping[remove_text_start] == 
              cur.text_token_mapping[remove_text_start-1]) {
      remove_text_start--;
    }
    // Same for the update
    int add_text_start = update_start + lcs_len;
    while (add_text_start > 0 && 
              update.text_token_mapping[add_text_start] == 
              update.text_token_mapping[add_text_start-1]) {
      add_text_start--;
    }


    // Delete tokens to the right
    if (remove_text_start < cur.text.size()) {
      printf("\tCur Delete: '%s'\n", cur.text.substr(remove_text_start).c_str());
      int id_right_remove = cur.text_token_mapping[remove_text_start];
      cur.tokens.erase(cur.tokens.begin() + id_right_remove, cur.tokens.end());
      cur.probs.erase(cur.probs.begin() + id_right_remove, cur.probs.end());
      cur.counts.erase(cur.counts.begin() + id_right_remove, cur.counts.end());

      // Remove text and mapping
      cur.text.erase(cur.text.begin() + remove_text_start, cur.text.end());
      cur.text_token_mapping.erase(cur.text_token_mapping.begin() + remove_text_start, 
                                        cur.text_token_mapping.end());
    }

    // Add tokens from the update
    std::vector<int> new_token_ids_right = update.get_token_ids_right(add_text_start);
    printf("\tCur Add: '%s'\n", update.text.substr(add_text_start).c_str());
    printf("\t\t Tokens: ");
    for (auto & new_id : new_token_ids_right) {
      std::string new_token = update.tokens[new_id];
      printf("'%s', ", new_token.c_str());
      for (int char_id = 0; char_id < new_token.size(); char_id++) {
        cur.text_token_mapping.push_back(cur.tokens.size());
      }
      cur.text += new_token;
      cur.tokens.push_back(new_token);
      cur.probs.push_back(update.probs[new_id]);
      cur.counts.push_back(update.counts[new_id]);
    }
    printf("\n");
    printf("  => '%s'\n", cur.text.c_str());
  }

  void merge(WhisperSegment & cur, WhisperSegment & update, 
                int cur_start, int update_start, int lcs_len) {

    if (lcs_len == 0) {
      return;
    }

    // Skip to end of cur/update lcs.
    //  TODO:  Update token counts
    int cur_idx = cur_start + lcs_len;
    int update_idx = update_start + lcs_len;
    // printf(" last same char / diff char: '%c / %c'\n", cur.text[cur_idx], update.text[update_idx]);

    
    int prev_token_idx = -1;

    // Increase the token reference and the count for each token in the lcs TODO
    // for (int i=0; i<lcs_len; i++) {
    //   auto cur_token_idx = cur.text_token_mapping[cur_start + i];
    //   if (prev_token_idx != cur_token_idx) {
    //     cur.counts[cur_token_idx]++;
    //     prev_token_idx = cur_token_idx;
    //   }
    // }

    printf("    CUR: '%s'\n", cur.text.c_str());
    printf(" UPDATE: '%s'\n", update.text.c_str());

    while (cur_idx < cur.text.size() && update_idx < update.text.size()) {
      auto cur_token_idx = cur.text_token_mapping[cur_idx];
      auto update_token_idx = update.text_token_mapping[update_idx];
      std::string cur_token_val = cur.tokens[cur_token_idx];
      std::string update_token_val = update.tokens[update_token_idx];
      printf("\t    CUR idx: %d\n", cur_idx);
      printf("\t UPDATE idx: %d\n", update_idx);
      printf("\t    CUR rem: '%s'\n", cur.text.substr(cur_idx).c_str());
      printf("\t UPDATE rem: '%s'\n", update.text.substr(update_idx).c_str());
      
      // printf("    CUR     text_idx->token: %d -> %d\n", cur_idx, cur_token_idx);
      // printf("    CUR      cur token text: '%s'\n", cur_token_val.c_str());
      // printf(" UPDATE     text_idx->token: %d -> %d\n", update_idx, update_token_idx);
      // printf(" UPDATE      new token text: '%s'\n", update_token_val.c_str());
      printf("\tReplacing:  '%s'->'%s'\n", cur_token_val.c_str(), update_token_val.c_str());


      // DEBUG -- assert equal
      std::string cur_token_str;
      int cur_idx_copy = cur_idx;
      while (cur.text_token_mapping[cur_idx_copy] == cur_token_idx) {
        cur_token_str.push_back(cur.text[cur_idx_copy++]);
      }
      if (cur_token_str != cur_token_val) {
        printf("\tERROR current string token: '%s' v.s. '%s'\n", cur_token_str.c_str(), cur_token_val.c_str());
      }
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
            cur.text[cur_idx+off] = update.text[update_idx+off];
          }
        } else if (cur_token_val.size() > update_token_val.size()) {
          // 1.2
          printf("- -1.2 cur token is larger\n");
          auto off = 0;
          for (; off < update_token_val.size(); off++) {
            cur.text[cur_idx+off] = update.text[update_idx+off];
          }
          cur.text.erase(cur_idx+off, cur_token_val.size()-off);
        } else {
          // 1.3
          printf("- -1.3 update token is larger\n");
          auto off = 0;
          for (; off < cur_token_val.size(); off++) {
            cur.text[cur_idx+off] = update.text[update_idx+off];
          }
          cur.text.insert(cur_idx+off, update_token_val.substr(off));
        }

        printf("- ----- REPLACED to : '%s'\n", cur.text.c_str());
        cur.add_and_swap_conflict(cur_idx, update, update_idx);

        // 3.
        // cur_idx += cur_token_val.size();
        cur_idx += update_token_val.size(); // cur text is equal to updated
        update_idx += update_token_val.size();
      }
    }

    // TODO:  Test removing extra from cur

    // Add what is remaining in update on to cur.text
    while (update_idx < update.text.size()) {
      int update_token_idx = update.text_token_mapping[update_idx];
      std::string update_token = update.tokens[update_token_idx];
      
      // Add token
      int new_token_idx = cur.tokens.size();
      cur.tokens.push_back(update_token);
      cur.probs.push_back(update.probs[update_token_idx]);
      cur.counts.push_back(update.counts[update_token_idx]);

      // Add text and mapping
      cur.text += update_token;
      for (auto i=0; i<update_token.size(); i++) {
        cur.text_token_mapping.push_back(new_token_idx);
      }

      update_idx += update_token.size();
      // cur.text += update.text.substr(update_idx);

    }
  }

  void merge(WhisperSegment & cur, WhisperSegment & update) {
    clobber(cur, update);
  }

  void clobber(WhisperSegment & cur, WhisperSegment & update) {
    // printf("       update: '%s'\n", update.text.c_str());
    // printf("       before: '%s'\n", cur.text.c_str());
    cur = update;
    // printf("        after: '%s'\n", cur.text.c_str());
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

    printf("New segments:  ");
    for (auto & seg : add_segments) {
      printf("'%s', ", seg.text.c_str());
    }
    printf("\n");

    // Iterate last to first
    for (int i=segments.size()-1; i>=0; i--) {
      if (segments_finished[i]) {
        // continue;
        break;
      } // If the segment does not get updated now, mark it as finished.
      bool updated = false;

      printf("old_segment: '%s'\n", segments[i].text.c_str());

      for (auto j=0; j<add_segments.size(); j++) {
        if (!add_segments_mask[j]) {
          continue; // New segment has already been merged/used to update
        }
        auto new_segment = add_segments[j];

        float overlap;
        int start_str1;
        int start_str2;
        int len;
        
        std::tie(overlap, start_str1, start_str2, len) = segments[i].text_overlap(new_segment);
        
        if (overlap > 0.4) {
          std::string lcs = segments[i].text.substr(start_str1, len);
          printf("   v.s. new: '%s'\n", new_segment.text.c_str());
          printf("\tlcs:  (%.4f % ):  '%s'\n", overlap, lcs.c_str());

          if (segments[i].text.size() <= new_segment.text.size()) {
            printf("\t!!-MERGED\n");
            // merge(segments[i], new_segment, start_str1, start_str2, len);
            merge_clobber_right(segments[i], new_segment, start_str1, start_str2, len);
            // clobber(segments[i], new_segment);

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
       -- Solution attempt:  return new added text.  Check if new added text is part of next sbstr, remove or combine
       -- Opiton 2: Combine transcripts?

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
