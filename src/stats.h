#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
inline double percentile(std::vector<double> values, double p) {
  if (values.empty() || p < 0 || p > 1) throw std::invalid_argument("invalid percentile");
  std::sort(values.begin(), values.end());
  double pos = p * (values.size() - 1);
  auto lo = static_cast<size_t>(pos), hi = static_cast<size_t>(std::ceil(pos));
  return values[lo] + (values[hi] - values[lo]) * (pos - lo);
}
