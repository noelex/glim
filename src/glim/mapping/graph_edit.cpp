#include <glim/mapping/graph_edit.hpp>

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include <gtsam_points/factors/linear_damping_factor.hpp>

namespace glim {
namespace {

bool is_pose_key(const gtsam::Key key) {
  return gtsam::Symbol(key).chr() == 'x';
}

class KeyComponents {
public:
  explicit KeyComponents(const gtsam::Values& values) {
    for (const auto& value : values) {
      parents[value.key] = value.key;
    }
  }

  gtsam::Key find(const gtsam::Key key) {
    auto& parent = parents.at(key);
    if (parent != key) {
      parent = find(parent);
    }
    return parent;
  }

  void connect(const gtsam::Key first, const gtsam::Key second) {
    const auto first_root = find(first);
    const auto second_root = find(second);
    if (first_root == second_root) {
      return;
    }

    // Always keep the smaller root so component output is deterministic.
    if (first_root < second_root) {
      parents[second_root] = first_root;
    } else {
      parents[first_root] = second_root;
    }
  }

private:
  std::map<gtsam::Key, gtsam::Key> parents;
};

}  // namespace

gtsam::KeyVector SubmapStateKeys::all() const {
  return {origin_pose, endpoint_poses[0], endpoint_poses[1], velocities[0], velocities[1], biases[0], biases[1]};
}

SubmapStateKeys submap_state_keys(const int submap_id) {
  if (submap_id < 0) {
    throw std::invalid_argument("submap ID must be nonnegative");
  }

  using gtsam::symbol_shorthand::B;
  using gtsam::symbol_shorthand::E;
  using gtsam::symbol_shorthand::V;
  using gtsam::symbol_shorthand::X;
  return {X(submap_id), {E(2 * submap_id), E(2 * submap_id + 1)}, {V(2 * submap_id), V(2 * submap_id + 1)}, {B(2 * submap_id), B(2 * submap_id + 1)}};
}

gtsam::KeySet collect_submap_state_keys(const std::vector<int>& submap_ids) {
  gtsam::KeySet keys;
  for (const int submap_id : submap_ids) {
    const auto state_keys = submap_state_keys(submap_id).all();
    keys.insert(state_keys.begin(), state_keys.end());
  }
  return keys;
}

gtsam::Values filter_values_by_keys(const gtsam::Values& values, const gtsam::KeySet& removed_keys) {
  gtsam::Values filtered;
  for (const auto& value : values) {
    if (!removed_keys.count(value.key)) {
      filtered.insert(value.key, value.value);
    }
  }
  return filtered;
}

gtsam::NonlinearFactorGraph filter_factors_by_keys(const gtsam::NonlinearFactorGraph& factors, const gtsam::KeySet& removed_keys) {
  gtsam::NonlinearFactorGraph filtered;
  for (const auto& factor : factors) {
    if (!factor) {
      continue;
    }

    // Inspect keys instead of factor types so future multi-key factors are
    // pruned without adding factor-specific cases here.
    const bool touches_removed_key = std::any_of(factor->keys().begin(), factor->keys().end(), [&](const gtsam::Key key) { return removed_keys.count(key) != 0; });
    if (!touches_removed_key) {
      filtered.push_back(factor);
    }
  }
  return filtered;
}

void validate_factor_keys(const gtsam::NonlinearFactorGraph& factors, const gtsam::Values& values) {
  for (size_t i = 0; i < factors.size(); i++) {
    const auto& factor = factors[i];
    if (!factor) {
      throw std::invalid_argument("factor " + std::to_string(i) + " is null");
    }
    for (const gtsam::Key key : factor->keys()) {
      if (!values.exists(key)) {
        throw std::invalid_argument("factor " + std::to_string(i) + " references missing key " + gtsam::DefaultKeyFormatter(key));
      }
    }
  }
}

std::vector<PoseGaugeAnchor> find_pose_gauge_anchors(const gtsam::NonlinearFactorGraph& factors) {
  std::vector<PoseGaugeAnchor> anchors;
  for (const auto& factor : factors) {
    if (!factor) {
      continue;
    }

    const auto prior = dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(factor.get());
    if (prior && is_pose_key(prior->key())) {
      anchors.push_back({PoseGaugeAnchor::Type::POSE_PRIOR, prior->key(), prior->prior(), prior->noiseModel(), {}});
      continue;
    }

    const auto damping = dynamic_cast<const gtsam_points::LinearDampingFactor*>(factor.get());
    if (damping && damping->keys().size() == 1 && is_pose_key(damping->keys().front())) {
      // LinearContainerFactor::dim() is always one for Hessian-backed factors,
      // so identify a full pose gauge anchor from the information matrix instead.
      const auto information = damping->toHessian()->information();
      if (information.rows() == 6 && information.cols() == 6) {
        const auto diagonal = information.diagonal();
        if ((diagonal.array() > 0.0).all()) {
          anchors.push_back({PoseGaugeAnchor::Type::LINEAR_DAMPING, damping->keys().front(), {}, nullptr, diagonal});
        }
      }
    }
  }
  return anchors;
}

gtsam::Key select_pose_gauge_anchor_key(const std::vector<PoseGaugeAnchor>& anchors) {
  if (anchors.empty()) {
    throw std::invalid_argument("no full pose gauge anchor found");
  }

  return std::min_element(anchors.begin(), anchors.end(), [](const auto& lhs, const auto& rhs) { return gtsam::Symbol(lhs.key).index() < gtsam::Symbol(rhs.key).index(); })->key;
}

void ensure_pose_gauge_anchor(
  gtsam::NonlinearFactorGraph& candidate_factors,
  const gtsam::Values& candidate_values,
  const std::vector<PoseGaugeAnchor>& original_anchors,
  const std::vector<int>& active_target_submaps) {
  const auto primary_key = select_pose_gauge_anchor_key(original_anchors);
  const auto candidate_anchors = find_pose_gauge_anchors(candidate_factors);
  const bool primary_anchor_exists = std::any_of(candidate_anchors.begin(), candidate_anchors.end(), [primary_key](const auto& anchor) { return anchor.key == primary_key; });
  if (primary_anchor_exists) {
    return;
  }
  if (candidate_values.exists(primary_key)) {
    throw std::invalid_argument("primary pose gauge anchor is missing while its value remains active");
  }
  if (active_target_submaps.empty()) {
    throw std::invalid_argument("cannot transfer pose gauge anchor without an active target submap");
  }

  for (const int submap_id : active_target_submaps) {
    if (!candidate_values.exists(submap_state_keys(submap_id).origin_pose)) {
      throw std::invalid_argument("active target submap is missing its pose value");
    }
  }

  const int target_id = *std::min_element(active_target_submaps.begin(), active_target_submaps.end());
  const auto target_key = submap_state_keys(target_id).origin_pose;
  for (const auto& anchor : original_anchors) {
    if (anchor.key != primary_key) {
      continue;
    }

    if (anchor.type == PoseGaugeAnchor::Type::POSE_PRIOR) {
      // GlobalMapping treats Pose3 priors as gauge constraints, so preserve
      // their strength while anchoring the new pose at its current estimate.
      candidate_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(target_key, candidate_values.at<gtsam::Pose3>(target_key), anchor.noise_model);
    } else {
      candidate_factors.emplace_shared<gtsam_points::LinearDampingFactor>(target_key, anchor.damping_diagonal);
    }
  }
}

GraphConnectivity analyze_graph_connectivity(const gtsam::Values& values, const gtsam::NonlinearFactorGraph& factors, const gtsam::Key anchor_key) {
  validate_factor_keys(factors, values);
  if (!values.exists(anchor_key)) {
    throw std::invalid_argument("pose gauge anchor key is missing from values");
  }

  KeyComponents key_components(values);
  for (const auto& factor : factors) {
    const auto& keys = factor->keys();
    // A factor is a hyperedge: joining every key to the first key connects the
    // whole hyperedge without reducing it to submap-owner X-X edges.
    for (size_t i = 1; i < keys.size(); i++) {
      key_components.connect(keys.front(), keys[i]);
    }
  }

  std::map<gtsam::Key, gtsam::KeyVector> components_by_root;
  for (const auto& value : values) {
    components_by_root[key_components.find(value.key)].push_back(value.key);
  }

  GraphConnectivity connectivity;
  for (auto& item : components_by_root) {
    connectivity.components.push_back(std::move(item.second));
  }

  const auto anchor_root = key_components.find(anchor_key);
  std::set<gtsam::Key> pose_roots;
  for (const auto& value : values) {
    const auto root = key_components.find(value.key);
    if (root == anchor_root) {
      connectivity.reachable_keys.push_back(value.key);
    }
    if (is_pose_key(value.key)) {
      pose_roots.insert(root);
      if (root != anchor_root) {
        connectivity.unreachable_pose_keys.push_back(value.key);
      }
    }
  }
  connectivity.pose_component_count = pose_roots.size();

  return connectivity;
}

gtsam::Values transform_session_values(const gtsam::Values& values, const gtsam::Pose3& transform) {
  gtsam::Values transformed;
  for (const auto& value : values) {
    const gtsam::Symbol symbol(value.key);
    if (symbol.chr() == 'x' || symbol.chr() == 'e') {
      transformed.insert(value.key, transform * value.value.cast<gtsam::Pose3>());
    } else if (symbol.chr() == 'v') {
      transformed.insert(value.key, transform.rotation().rotate(value.value.cast<gtsam::Vector3>()));
    } else {
      transformed.insert(value.key, value.value);
    }
  }
  return transformed;
}

TrialISAM2Build build_trial_isam2(const gtsam::NonlinearFactorGraph& factors, const gtsam::Values& values, const gtsam::ISAM2Params& params) {
  validate_factor_keys(factors, values);

  // iSAM2 can retain an unreferenced initial value without proving it is
  // observable, so reject such values before accepting the trial build.
  gtsam::KeySet referenced_keys;
  for (const auto& factor : factors) {
    referenced_keys.insert(factor->keys().begin(), factor->keys().end());
  }
  for (const auto& value : values) {
    if (!referenced_keys.count(value.key)) {
      throw std::invalid_argument("candidate value is not referenced by any factor: " + gtsam::DefaultKeyFormatter(value.key));
    }
  }

  const auto anchor_key = select_pose_gauge_anchor_key(find_pose_gauge_anchors(factors));
  const auto connectivity = analyze_graph_connectivity(values, factors, anchor_key);
  if (!connectivity.all_poses_reachable()) {
    throw std::invalid_argument("not all pose values are reachable from the pose gauge anchor");
  }

  TrialISAM2Build result;
  result.optimizer = std::make_unique<gtsam_points::ISAM2Ext>(params);
  result.update_result = result.optimizer->update(factors, values);
  result.estimate = result.optimizer->calculateEstimate();

  if (result.estimate.size() != values.size()) {
    throw std::runtime_error("trial iSAM2 result does not contain all candidate values");
  }
  for (const auto& value : values) {
    if (!result.estimate.exists(value.key)) {
      throw std::runtime_error("trial iSAM2 result is missing key " + gtsam::DefaultKeyFormatter(value.key));
    }
  }

  return result;
}

}  // namespace glim
