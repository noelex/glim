#include <glim/mapping/async_global_mapping.hpp>
#include <glim/mapping/callbacks.hpp>

#include <memory>
#include <stdexcept>
#include <string>

namespace {

void expect_callbacks(const bool expected) {
  using glim::GlobalMappingCallbacks;
  const auto expect = [=](const bool value, const char* name) {
    if (value != expected) {
      throw std::runtime_error(std::string("unexpected callback lifetime: ") + name);
    }
  };
  expect(GlobalMappingCallbacks::request_to_optimize, "optimize");
  expect(GlobalMappingCallbacks::request_to_add_graph_factors, "add_graph_factors");
  expect(GlobalMappingCallbacks::request_to_merge_sessions, "merge_sessions");
  expect(GlobalMappingCallbacks::request_to_recover, "recover");
  expect(GlobalMappingCallbacks::request_to_find_overlapping_submaps, "find_overlapping_submaps");
}

}  // namespace

int main() {
  expect_callbacks(false);
  {
    auto mapping = std::make_shared<glim::GlobalMappingBase>();
    glim::AsyncGlobalMapping async_mapping(mapping, 1000000);
    expect_callbacks(true);
  }
  expect_callbacks(false);
  return 0;
}
