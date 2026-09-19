#include <glim/mapping/graph_metadata.hpp>

#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using glim::GraphMetadata;
using glim::MatchingCostRecord;
using glim::SubmapRange;

void expect(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Exception = std::exception>
void expect_throw(const std::function<void()>& function, const std::string& message) {
  try {
    function();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(message);
}

GraphMetadata parse(const std::string& text) {
  std::istringstream stream(text);
  return glim::parse_graph_metadata(stream);
}

void test_ranges() {
  const std::vector<SubmapRange> input{{8, 9}, {1, 3}, {5, 7}, {2, 4}, {12, 12}};
  const std::vector<SubmapRange> expected{{1, 9}, {12, 12}};
  expect(glim::normalize_submap_ranges(input) == expected, "normalize must sort and merge overlapping or adjacent ranges");

  expect_throw<std::invalid_argument>([] { glim::normalize_submap_ranges({{4, 3}}); }, "inverted range must be rejected");
  expect_throw<std::invalid_argument>([] { glim::normalize_submap_ranges({{-1, 3}}); }, "negative range must be rejected");

  const std::vector<SubmapRange> united{{1, 5}, {8, 12}};
  expect(glim::union_submap_ranges({{1, 3}, {8, 9}}, {{4, 5}, {10, 12}}) == united, "range union is incorrect");

  const std::vector<SubmapRange> subtracted{{3, 3}, {6, 7}, {23, 23}};
  expect(glim::subtract_submap_ranges({{1, 10}, {20, 25}}, {{0, 2}, {4, 5}, {8, 22}, {24, 30}}) == subtracted, "range subtraction is incorrect");

  expect(glim::offset_submap_ranges({{1, 2}, {5, 8}}, 10) == std::vector<SubmapRange>({{11, 12}, {15, 18}}), "range offset is incorrect");
  expect_throw<std::out_of_range>([] { glim::offset_submap_ranges({{0, 2}}, -1); }, "negative offset result must be rejected");

  const std::vector<uint8_t> mask{0, 1, 1, 1, 0, 1, 0};
  expect(glim::submap_ranges_to_mask({{1, 3}, {5, 5}}, 7) == mask, "range-to-mask conversion is incorrect");
  expect(glim::submap_mask_to_ranges(mask) == std::vector<SubmapRange>({{1, 3}, {5, 5}}), "mask-to-range conversion is incorrect");
  expect(glim::submap_ranges_to_ids({{1, 3}, {5, 5}}) == std::vector<int>({1, 2, 3, 5}), "range-to-ID conversion is incorrect");
  expect(glim::submap_ids_to_ranges({5, 2, 1, 3, 2}) == std::vector<SubmapRange>({{1, 3}, {5, 5}}), "ID-to-range conversion is incorrect");
  expect_throw<std::invalid_argument>([] { glim::submap_ids_to_ranges({1, -1}); }, "negative submap ID must be rejected");
  expect(glim::count_submaps({{1, 3}, {5, 5}}) == 4, "range count is incorrect");
  expect(glim::is_submap_pruned(mask, 2), "pruned mask lookup failed");
  expect(!glim::is_submap_pruned(mask, 4), "active mask lookup failed");
  expect(!glim::is_submap_pruned(mask, 20), "out-of-range mask lookup must be false");
  expect_throw<std::out_of_range>([] { glim::submap_ranges_to_mask({{4, 5}}, 5); }, "out-of-bounds range must be rejected");
}

void test_legacy_metadata() {
  const auto metadata = parse(
    "num_submaps: 4\n"
    "num_all_frames: 20\n"
    "num_matching_cost_factors: 2\n"
    "matching_cost vgicp 0 1\n"
    "matching_cost vgicp_gpu 2 3\n");

  expect(metadata.version == 0, "legacy metadata version is incorrect");
  expect(metadata.num_submaps == 4, "legacy num_submaps is incorrect");
  expect(metadata.num_active_frames == 20, "legacy active frame count must equal all frames");
  expect(metadata.pruned_ranges.empty(), "legacy metadata must not contain pruned ranges");
  expect(metadata.matching_cost_factors.size() == 2, "legacy matching cost records are missing");
}

void test_v1_round_trip() {
  GraphMetadata input;
  input.num_submaps = 10;
  input.num_all_frames = 80;
  input.num_active_frames = 55;
  input.pruned_ranges = {{7, 7}, {2, 3}};
  input.matching_cost_factors = {{"vgicp", 0, 1}, {"vgicp_gpu", 4, 9}};

  std::ostringstream output;
  glim::write_graph_metadata(output, input);
  const std::string expected =
    "graph_metadata_version: 1\n"
    "num_submaps: 10\n"
    "num_all_frames: 80\n"
    "num_active_frames: 55\n"
    "num_pruned_submaps: 3\n"
    "num_pruned_ranges: 2\n"
    "pruned_range 2 3\n"
    "pruned_range 7 7\n"
    "num_matching_cost_factors: 2\n"
    "matching_cost vgicp 0 1\n"
    "matching_cost vgicp_gpu 4 9\n";
  expect(output.str() == expected, "v1 metadata records were not written in canonical order");
  const auto parsed = parse(output.str());

  expect(parsed.version == 1, "v1 metadata version is incorrect");
  expect(parsed.num_submaps == input.num_submaps, "v1 num_submaps changed during round trip");
  expect(parsed.num_all_frames == input.num_all_frames, "v1 num_all_frames changed during round trip");
  expect(parsed.num_active_frames == input.num_active_frames, "v1 num_active_frames changed during round trip");
  expect(parsed.pruned_ranges == std::vector<SubmapRange>({{2, 3}, {7, 7}}), "v1 ranges were not canonicalized");
  expect(parsed.matching_cost_factors == input.matching_cost_factors, "v1 matching records changed during round trip");

  const auto with_unknown_record = parse(
    "graph_metadata_version: 1\n"
    "num_submaps: 2\n"
    "future_record any values are skipped\n"
    "num_all_frames: 4\n"
    "num_active_frames: 4\n"
    "num_matching_cost_factors: 0\n"
    "num_pruned_submaps: 0\n"
    "num_pruned_ranges: 0\n");
  expect(with_unknown_record.num_submaps == 2, "unknown v1 record must not affect known records");
}

void test_invalid_metadata() {
  expect_throw([] { parse(""); }, "empty metadata must be rejected");
  expect_throw([] { parse("graph_metadata_version: 2\n"); }, "unknown metadata version must be rejected");
  expect_throw([] { parse("num_submaps: 2\nnum_all_frames: 4\n"); }, "truncated legacy metadata must be rejected");
  expect_throw([] { parse("num_submaps: 2\nnum_all_frames: 4\nnum_matching_cost_factors: 1\nmatching_cost vgicp 0\n"); }, "truncated matching record must be rejected");
  expect_throw([] { parse("num_submaps: 2 extra\nnum_all_frames: 4\nnum_matching_cost_factors: 0\n"); }, "extra values must be rejected");
  expect_throw(
    [] {
      parse(
        "num_submaps: 2\n"
        "num_all_frames: 4\n"
        "num_matching_cost_factors: 1\n");
    },
    "matching cost count mismatch must be rejected");
  expect_throw(
    [] {
      parse(
        "graph_metadata_version: 1\n"
        "num_submaps: 5\n"
        "num_all_frames: 10\n"
        "num_active_frames: 8\n"
        "num_matching_cost_factors: 0\n"
        "num_pruned_submaps: 2\n"
        "num_pruned_ranges: 1\n"
        "pruned_range 4 5\n");
    },
    "out-of-bounds pruned range must be rejected");
  expect_throw(
    [] {
      parse(
        "graph_metadata_version: 1\n"
        "num_submaps: 5\n"
        "num_all_frames: 10\n"
        "num_active_frames: 8\n"
        "num_matching_cost_factors: 0\n"
        "num_pruned_submaps: 1\n"
        "num_pruned_ranges: 2\n"
        "pruned_range 2 2\n");
    },
    "pruned range count mismatch must be rejected");
  expect_throw(
    [] {
      parse(
        "graph_metadata_version: 1\n"
        "num_submaps: 5\n"
        "num_all_frames: 10\n"
        "num_active_frames: 8\n"
        "num_matching_cost_factors: 0\n"
        "num_pruned_submaps: 2\n"
        "num_pruned_ranges: 1\n"
        "pruned_range 2 2\n");
    },
    "pruned submap count mismatch must be rejected");
  expect_throw(
    [] {
      parse(
        "graph_metadata_version: 1\n"
        "num_submaps: 5\n"
        "num_all_frames: 10\n"
        "num_active_frames: 8\n"
        "num_matching_cost_factors: 0\n"
        "num_pruned_submaps: 2\n"
        "num_pruned_ranges: 2\n"
        "pruned_range 1 1\n"
        "pruned_range 2 2\n");
    },
    "adjacent noncanonical ranges must be rejected");
  expect_throw(
    [] {
      parse(
        "graph_metadata_version: 1\n"
        "num_submaps: 5\n"
        "num_all_frames: 10\n"
        "num_active_frames: 8\n"
        "num_matching_cost_factors: 1\n"
        "num_pruned_submaps: 1\n"
        "num_pruned_ranges: 1\n"
        "pruned_range 2 2\n"
        "matching_cost vgicp 1 2\n");
    },
    "matching record referencing a pruned submap must be rejected");
  expect_throw(
    [] {
      parse(
        "graph_metadata_version: 1\n"
        "num_submaps: 5\n"
        "num_all_frames: 10\n"
        "num_active_frames: 10\n"
        "num_matching_cost_factors: 1\n"
        "num_pruned_submaps: 0\n"
        "num_pruned_ranges: 0\n"
        "matching_cost vgicp 1 5\n");
    },
    "matching record referencing a missing submap must be rejected");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    test_ranges();
    test_legacy_metadata();
    test_v1_round_trip();
    test_invalid_metadata();

    for (int i = 1; i < argc; i++) {
      std::ifstream stream(argv[i]);
      expect(static_cast<bool>(stream), "failed to open metadata fixture " + std::string(argv[i]));
      glim::parse_graph_metadata(stream);
    }
  } catch (const std::exception& e) {
    std::cerr << "graph_metadata_test failed: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
