#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace glim {

struct SubmapRange {
  int first;
  int last;

  bool operator==(const SubmapRange& other) const { return first == other.first && last == other.last; }
  bool operator!=(const SubmapRange& other) const { return !(*this == other); }
};

std::vector<SubmapRange> normalize_submap_ranges(const std::vector<SubmapRange>& ranges);
std::vector<SubmapRange> union_submap_ranges(const std::vector<SubmapRange>& lhs, const std::vector<SubmapRange>& rhs);
std::vector<SubmapRange> subtract_submap_ranges(const std::vector<SubmapRange>& ranges, const std::vector<SubmapRange>& removed);
std::vector<SubmapRange> offset_submap_ranges(const std::vector<SubmapRange>& ranges, int offset);

std::vector<uint8_t> submap_ranges_to_mask(const std::vector<SubmapRange>& ranges, int num_submaps);
std::vector<SubmapRange> submap_mask_to_ranges(const std::vector<uint8_t>& mask);
std::vector<int> submap_ranges_to_ids(const std::vector<SubmapRange>& ranges);
std::vector<SubmapRange> submap_ids_to_ranges(const std::vector<int>& ids);
int count_submaps(const std::vector<SubmapRange>& ranges);
bool is_submap_pruned(const std::vector<uint8_t>& mask, int submap_id);

struct MatchingCostRecord {
  std::string type;
  int first;
  int second;

  bool operator==(const MatchingCostRecord& other) const { return type == other.type && first == other.first && second == other.second; }
};

struct GraphMetadata {
  int version = 1;
  int num_submaps = 0;
  int num_all_frames = 0;
  int num_active_frames = 0;
  std::vector<SubmapRange> pruned_ranges;
  std::vector<MatchingCostRecord> matching_cost_factors;
};

GraphMetadata parse_graph_metadata(std::istream& stream);
void write_graph_metadata(std::ostream& stream, const GraphMetadata& metadata);

}  // namespace glim
