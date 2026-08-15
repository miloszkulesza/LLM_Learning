#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>

int main() {
    const std::uint64_t iterations = 200'000'000;
    const double param1 = 4.0;
    const double param2 = 1.0;
    double result = 1.0;

    auto start = std::chrono::high_resolution_clock::now();

    for (std::uint64_t i = 1; i <= iterations; ++i) {
        double j = i * param1 - param2;
        result -= 1.0 / j;
        j = i * param1 + param2;
        result += 1.0 / j;
    }

    result *= 4.0;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(12) << "Result: " << result << '\n';
    std::cout << std::setprecision(6) << "Execution Time: " << elapsed.count() << " seconds\n";

    return 0;
}
