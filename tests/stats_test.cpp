#include "stats.h"
#include <cassert>
#include <cmath>
#include <stdexcept>
int main() {
  assert(std::abs(percentile({4, 1, 3, 2}, .50) - 2.5) < 1e-9);
  assert(std::abs(percentile({4, 1, 3, 2}, .95) - 3.85) < 1e-9);
  bool rejected = false;
  try { percentile({}, .5); } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}
