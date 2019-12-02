// before C++11
#include <iostream>
#include <vector>
#include <string>
// C++11
#include <regex>
// C
#include <string.h>

using std::vector;
using std::string;

// refer to: https://www.zhihu.com/question/36642771/answer/865135551

// before C++11
void split(const string &s, vector<string> &tokens, const string &delimiters = " ") {
    string::size_type lastPos = s.find_first_not_of(delimiters, 0);
    string::size_type pos = s.find_first_of(delimiters, lastPos);
    while (string::npos != pos || string::npos != lastPos) {
        tokens.push_back(s.substr(lastPos, pos - lastPos));  //use emplace_back after C++11
        lastPos = s.find_first_not_of(delimiters, pos);
        pos = s.find_first_of(delimiters, lastPos);
    }
}


int main() {
    // before C++11
    string str = "hello world";
    vector<string> tokens;
    split(str, tokens);
    for (auto &s:tokens) {
        std::cout << s << "\n";
    }

    // C++11, regex
    std::regex ws_re("\\s+"); // whitespace
    std::vector<std::string> v(std::sregex_token_iterator(str.begin(), str.end(), ws_re, -1),
                               std::sregex_token_iterator());
    for (auto &&s: v)
        std::cout << s << "\n";

    // No verification
    // C style
    char *token = std::strtok(str.data(), " ");
    while (token != NULL) {
        printf("%s\n", token);
        token = strtok(NULL, " ");
    }

}