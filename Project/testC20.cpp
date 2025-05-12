#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> numbers = {3, 1, 4, 1, 5, 9, 2, 6};
    std::ranges::sort(numbers);
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    return 0;
}