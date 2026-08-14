
#include <iostream>
#include <chrono>

double calculate(int iterations, double param1, double param2) {
    double result = 1.0;
    for (int i = 1; i <= iterations; ++i) {
        double j = i * param1 - param2;
        result -= (1 / j);
        j = i * param1 + param2;
        result += (1 / j);
    }
    return result;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    double result = calculate(200000000, 4, 1) * 4;
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "Result: " << result << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}
