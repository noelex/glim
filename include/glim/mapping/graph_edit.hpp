#pragma once

#include <array>
#include <memory>
#include <vector>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam_points/optimizers/isam2_ext.hpp>
#include <gtsam_points/optimizers/isam2_result_ext.hpp>

namespace glim {

/**
 * @brief Keys owned by one submap
 *
 * A submap owns one origin pose and two endpoint extrinsic, velocity, and bias states.
 */
struct SubmapStateKeys {
  gtsam::Key pose;                         ///< Origin pose X(i)
  std::array<gtsam::Key, 2> extrinsics;    ///< Endpoint extrinsics E(2i), E(2i+1)
  std::array<gtsam::Key, 2> velocities;    ///< Endpoint velocities V(2i), V(2i+1)
  std::array<gtsam::Key, 2> biases;        ///< Endpoint biases B(2i), B(2i+1)

  /**
   * @brief Get all keys in the state block
   * @return Keys in X/E/V/B order
   */
  gtsam::KeyVector all() const;
};

/**
 * @brief Get the complete state keys owned by a submap
 * @param submap_id Submap ID
 * @return Complete X/E/V/B key block
 */
SubmapStateKeys submap_state_keys(int submap_id);

/**
 * @brief Collect complete state keys for multiple submaps
 * @param submap_ids Submap IDs
 * @return Deduplicated X/E/V/B keys
 */
gtsam::KeySet submap_state_key_set(const std::vector<int>& submap_ids);

/**
 * @brief Copy values except those in the removal set
 * @param values Input values
 * @param removed_keys Keys to remove
 * @return Filtered values
 */
gtsam::Values filter_values_by_keys(const gtsam::Values& values, const gtsam::KeySet& removed_keys);

/**
 * @brief Copy factors that do not touch any removed key
 * @param factors Input factors
 * @param removed_keys Keys to remove
 * @return Filtered factors
 */
gtsam::NonlinearFactorGraph filter_factors_by_keys(const gtsam::NonlinearFactorGraph& factors, const gtsam::KeySet& removed_keys);

/**
 * @brief Validate that every factor is non-null and references existing values
 * @param factors Factors to validate
 * @param values Values available to the factors
 * @throws std::invalid_argument if a factor is null or references a missing key
 */
void validate_factor_keys(const gtsam::NonlinearFactorGraph& factors, const gtsam::Values& values);

/**
 * @brief Full pose gauge anchor information
 */
struct PoseAnchor {
  gtsam::Key key;                       ///< Anchored X key
  gtsam::Pose3 prior;                   ///< Prior mean
  gtsam::SharedNoiseModel noise_model;  ///< Full pose prior noise model
};

/**
 * @brief Find full pose gauge anchors
 * @param factors Factors to inspect
 * @return PriorFactor<Pose3> records on X keys
 */
std::vector<PoseAnchor> find_full_pose_anchors(const gtsam::NonlinearFactorGraph& factors);

/**
 * @brief Find the unique full pose gauge anchor
 * @param factors Factors to inspect
 * @return Unique full pose anchor
 * @throws std::invalid_argument if the graph has zero or multiple anchors
 */
PoseAnchor find_unique_full_pose_anchor(const gtsam::NonlinearFactorGraph& factors);

/**
 * @brief Preserve or deterministically transfer the full pose gauge anchor
 * @param candidate_factors Candidate factors to update
 * @param candidate_values Candidate values after pruning
 * @param original_anchor Anchor from the stable graph
 * @param active_target_submaps Active target submap IDs
 *
 * If the original anchor was pruned, a new prior with the same noise model is
 * placed at the current pose of the lowest active target submap ID.
 */
void ensure_pose_anchor(
  gtsam::NonlinearFactorGraph& candidate_factors,
  const gtsam::Values& candidate_values,
  const PoseAnchor& original_anchor,
  const std::vector<int>& active_target_submaps);

/**
 * @brief Connectivity result for the complete key-factor graph
 */
struct KeyFactorConnectivity {
  std::vector<gtsam::KeyVector> components;  ///< Components containing all value keys
  gtsam::KeyVector reachable_keys;           ///< Keys reachable from the gauge anchor
  gtsam::KeyVector unreachable_pose_keys;    ///< X keys not reachable from the anchor
  size_t pose_component_count = 0;           ///< Number of components containing X keys

  /// @brief Check whether every pose is reachable from the gauge anchor
  bool all_poses_reachable() const { return unreachable_pose_keys.empty(); }
};

/**
 * @brief Analyze connectivity using factors as hyperedges over all value keys
 * @param values Graph values
 * @param factors Graph factors
 * @param anchor_key Full pose gauge anchor key
 * @return Complete key components and pose reachability diagnostics
 */
KeyFactorConnectivity analyze_key_factor_connectivity(
  const gtsam::Values& values,
  const gtsam::NonlinearFactorGraph& factors,
  gtsam::Key anchor_key);

/**
 * @brief Result of a successful strict temporary iSAM2 build
 */
struct StrictTrialBuildResult {
  std::unique_ptr<gtsam_points::ISAM2Ext> isam2;  ///< Validated temporary optimizer
  gtsam_points::ISAM2ResultExt update_result;     ///< Initial update result
  gtsam::Values estimate;                        ///< Complete optimized estimate
};

/**
 * @brief Build and validate a graph in a fresh iSAM2 instance
 * @param factors Complete candidate factors
 * @param values Complete candidate initial values
 * @param params Parameters shared with the formal optimizer
 * @return Temporary optimizer, update result, and complete estimate
 *
 * Exceptions are intentionally propagated to the transaction caller. The
 * stable optimizer is never accessed or modified by this function.
 */
StrictTrialBuildResult strict_trial_isam2_build(
  const gtsam::NonlinearFactorGraph& factors,
  const gtsam::Values& values,
  const gtsam::ISAM2Params& params);

}  // namespace glim
