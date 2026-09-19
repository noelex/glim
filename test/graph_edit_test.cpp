#include <glim/mapping/graph_edit.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/sam/RangeFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include <gtsam_points/factors/linear_damping_factor.hpp>

namespace {

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::E;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

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

class KeyOnlyFactor : public gtsam::NonlinearFactor {
public:
  explicit KeyOnlyFactor(const gtsam::KeyVector& keys) : gtsam::NonlinearFactor(keys) {}

  size_t dim() const override { return 0; }
  std::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values&) const override { return nullptr; }
  shared_ptr clone() const override { return std::make_shared<KeyOnlyFactor>(*this); }
};

void insert_submap_values(gtsam::Values& values, const int submap_id) {
  const auto keys = glim::submap_state_keys(submap_id);
  values.insert(keys.origin_pose, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(submap_id, 0.0, 0.0)));
  values.insert(keys.endpoint_poses[0], gtsam::Pose3());
  values.insert(keys.endpoint_poses[1], gtsam::Pose3());
  values.insert(keys.velocities[0], gtsam::Vector3(0.0, 0.0, 0.0));
  values.insert(keys.velocities[1], gtsam::Vector3(0.0, 0.0, 0.0));
  values.insert(keys.biases[0], gtsam::imuBias::ConstantBias());
  values.insert(keys.biases[1], gtsam::imuBias::ConstantBias());
}

void test_submap_keys_and_filtering() {
  const auto keys = glim::submap_state_keys(3);
  expect(keys.origin_pose == X(3), "submap origin pose key is incorrect");
  expect(keys.endpoint_poses == std::array<gtsam::Key, 2>{E(6), E(7)}, "submap endpoint pose keys are incorrect");
  expect(keys.velocities == std::array<gtsam::Key, 2>{V(6), V(7)}, "submap velocity keys are incorrect");
  expect(keys.biases == std::array<gtsam::Key, 2>{B(6), B(7)}, "submap bias keys are incorrect");
  expect(glim::collect_submap_state_keys({1, 1, 2}).size() == 14, "collected submap state keys must remove duplicate IDs");
  expect_throw<std::invalid_argument>([] { glim::submap_state_keys(-1); }, "negative submap ID must be rejected");

  gtsam::Values values;
  insert_submap_values(values, 0);
  insert_submap_values(values, 1);

  const auto noise6 = gtsam::noiseModel::Isotropic::Sigma(6, 1.0);
  const auto noise3 = gtsam::noiseModel::Isotropic::Sigma(3, 1.0);
  gtsam::NonlinearFactorGraph factors;
  factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), gtsam::Pose3(), noise6);
  factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(0), X(1), gtsam::Pose3(), noise6);
  factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(E(0), E(1), gtsam::Pose3(), noise6);
  factors.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(0), gtsam::Vector3::Zero(), noise3);
  factors.push_back(std::make_shared<KeyOnlyFactor>(gtsam::KeyVector{X(0), V(2), B(3)}));
  factors.push_back(gtsam::NonlinearFactor::shared_ptr());

  const auto removed_keys = glim::collect_submap_state_keys({1});
  const auto filtered_values = glim::filter_values_by_keys(values, removed_keys);
  const auto filtered_factors = glim::filter_factors_by_keys(factors, removed_keys);

  expect(filtered_values.size() == 7, "filtering one submap must remove its complete state block");
  for (const auto key : removed_keys) {
    expect(!filtered_values.exists(key), "filtered values still contain a removed key");
  }
  expect(filtered_factors.size() == 3, "all factors touching removed keys and null factors must be removed");
  for (const auto& factor : filtered_factors) {
    for (const auto key : factor->keys()) {
      expect(!removed_keys.count(key), "filtered factor still touches a removed key");
    }
  }
}

void test_filter_positions() {
  gtsam::Values values;
  values.insert(X(0), gtsam::Pose3());
  values.insert(X(1), gtsam::Pose3());
  values.insert(X(2), gtsam::Pose3());

  const auto noise = gtsam::noiseModel::Isotropic::Sigma(6, 1.0);
  gtsam::NonlinearFactorGraph factors;
  factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), gtsam::Pose3(), noise);
  factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(0), X(1), gtsam::Pose3(), noise);
  factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(1), X(2), gtsam::Pose3(), noise);

  const auto prune_first = glim::filter_factors_by_keys(factors, {X(0)});
  const auto prune_middle = glim::filter_factors_by_keys(factors, {X(1)});
  const auto prune_last = glim::filter_factors_by_keys(factors, {X(2)});
  expect(prune_first.size() == 1 && prune_first.front()->keys() == gtsam::KeyVector({X(1), X(2)}), "pruning the first submap produced incorrect factors");
  expect(prune_middle.size() == 1 && prune_middle.front()->keys() == gtsam::KeyVector({X(0)}), "pruning the middle submap produced incorrect factors");
  expect(prune_last.size() == 2, "pruning the last submap produced incorrect factors");
}

void test_pose_anchors() {
  const auto anchor_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.1, 0.2, 0.3, 1.0, 2.0, 3.0).finished());
  const gtsam::Pose3 original_prior(gtsam::Rot3::RzRyRx(0.1, 0.2, 0.3), gtsam::Point3(1.0, 2.0, 3.0));

  gtsam::NonlinearFactorGraph original_factors;
  original_factors.emplace_shared<gtsam_points::LinearDampingFactor>(X(2), 6, 456.0);
  original_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), original_prior, anchor_noise);
  original_factors.emplace_shared<gtsam_points::LinearDampingFactor>(X(0), 6, 123.0);
  original_factors.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(0), gtsam::Vector3::Zero(), gtsam::noiseModel::Isotropic::Sigma(3, 1.0));

  const auto original_anchors = glim::find_pose_gauge_anchors(original_factors);
  expect(original_anchors.size() == 3, "full pose anchors were not detected");
  expect(glim::select_pose_gauge_anchor_key(original_anchors) == X(0), "lowest anchored pose must be the primary gauge pose");
  expect_throw<std::invalid_argument>([] { glim::select_pose_gauge_anchor_key({}); }, "missing pose gauge anchors must be rejected");

  gtsam::NonlinearFactorGraph retained_factors = original_factors;
  gtsam::Values retained_values;
  retained_values.insert(X(0), gtsam::Pose3());
  glim::ensure_pose_gauge_anchor(retained_factors, retained_values, original_anchors, {0});
  expect(retained_factors.size() == original_factors.size(), "an active primary gauge pose must not gain duplicate anchors");

  gtsam::Values candidate_values;
  const gtsam::Pose3 target_pose(gtsam::Rot3::Rz(0.4), gtsam::Point3(5.0, 6.0, 7.0));
  candidate_values.insert(X(1), target_pose);
  candidate_values.insert(X(2), gtsam::Pose3());
  gtsam::NonlinearFactorGraph candidate_factors;
  candidate_factors.emplace_shared<gtsam_points::LinearDampingFactor>(X(2), 6, 456.0);
  glim::ensure_pose_gauge_anchor(candidate_factors, candidate_values, original_anchors, {2, 1});

  const auto transferred = glim::find_pose_gauge_anchors(candidate_factors);
  expect(transferred.size() == 3, "all primary anchors must transfer while recovery anchors remain in place");
  expect(glim::select_pose_gauge_anchor_key(transferred) == X(1), "gauge anchors must transfer to the lowest active target submap ID");

  const auto transferred_prior =
    std::find_if(transferred.begin(), transferred.end(), [](const auto& anchor) { return anchor.key == X(1) && anchor.type == glim::PoseGaugeAnchor::Type::POSE_PRIOR; });
  expect(transferred_prior != transferred.end(), "pose prior gauge anchor was not transferred");
  expect(transferred_prior->prior.equals(target_pose), "transferred gauge prior must use the current global pose");
  expect(transferred_prior->noise_model->equals(*anchor_noise), "transferred gauge prior must preserve its noise model");

  const auto transferred_damping =
    std::find_if(transferred.begin(), transferred.end(), [](const auto& anchor) { return anchor.key == X(1) && anchor.type == glim::PoseGaugeAnchor::Type::LINEAR_DAMPING; });
  expect(transferred_damping != transferred.end(), "linear damping gauge anchor was not transferred");
  expect(transferred_damping->damping_diagonal.isApprox(gtsam::Vector6::Constant(123.0)), "transferred damping semantics changed");

  const auto recovery_damping =
    std::find_if(transferred.begin(), transferred.end(), [](const auto& anchor) { return anchor.key == X(2) && anchor.type == glim::PoseGaugeAnchor::Type::LINEAR_DAMPING; });
  expect(recovery_damping != transferred.end(), "recovery damping was not preserved");
  expect(recovery_damping->damping_diagonal.isApprox(gtsam::Vector6::Constant(456.0)), "recovery damping semantics changed");

  gtsam::NonlinearFactorGraph second_candidate = glim::filter_factors_by_keys(candidate_factors, {X(1)});
  gtsam::Values second_values;
  second_values.insert(X(2), gtsam::Pose3());
  glim::ensure_pose_gauge_anchor(second_candidate, second_values, transferred, {2});
  const auto second_transfer = glim::find_pose_gauge_anchors(second_candidate);
  expect(second_transfer.size() == 3, "repeated gauge transfer lost anchors");
  expect(glim::select_pose_gauge_anchor_key(second_transfer) == X(2), "repeated pruning selected an incorrect primary gauge pose");
}

void test_connectivity() {
  gtsam::Values values;
  values.insert(X(0), gtsam::Pose3());
  values.insert(X(1), gtsam::Pose3());
  values.insert(X(2), gtsam::Pose3());
  values.insert(E(0), gtsam::Pose3());
  values.insert(V(0), gtsam::Vector3(0.0, 0.0, 0.0));
  values.insert(B(0), gtsam::imuBias::ConstantBias());

  const auto noise = gtsam::noiseModel::Isotropic::Sigma(6, 1.0);
  gtsam::NonlinearFactorGraph factors;
  factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), gtsam::Pose3(), noise);
  factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(0), E(0), gtsam::Pose3(), noise);
  factors.push_back(std::make_shared<KeyOnlyFactor>(gtsam::KeyVector{E(0), V(0), X(1)}));

  const auto connectivity = glim::analyze_graph_connectivity(values, factors, X(0));
  expect(connectivity.components.size() == 3, "full key-factor component count is incorrect");
  expect(connectivity.pose_component_count == 2, "pose component count is incorrect");
  expect(connectivity.unreachable_pose_keys == gtsam::KeyVector{X(2)}, "unreachable pose detection is incorrect");

  gtsam::Values without_x2;
  for (const auto& value : values) {
    if (value.key != X(2)) {
      without_x2.insert(value.key, value.value);
    }
  }
  const auto isolated_auxiliary = glim::analyze_graph_connectivity(without_x2, factors, X(0));
  expect(isolated_auxiliary.all_poses_reachable(), "isolated E/V/B state must not fail pose connectivity");
  expect(isolated_auxiliary.components.size() == 2, "isolated auxiliary state must remain visible in full components");

  gtsam::NonlinearFactorGraph missing_key_factor;
  missing_key_factor.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(0), X(9), gtsam::Pose3(), noise);
  expect_throw<std::invalid_argument>([&] { glim::analyze_graph_connectivity(values, missing_key_factor, X(0)); }, "factor references to missing keys must be rejected");
}

void test_trial_isam2_build() {
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(6, 0.1);
  gtsam::Values values;
  values.insert(X(0), gtsam::Pose3());
  values.insert(X(1), gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0)));

  gtsam::NonlinearFactorGraph factors;
  factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), gtsam::Pose3(), noise);
  factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(0), X(1), gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0)), noise);

  const auto result = glim::build_trial_isam2(factors, values, gtsam::ISAM2Params());
  expect(result.optimizer != nullptr, "trial build did not return the temporary iSAM2 instance");
  expect(result.estimate.size() == values.size(), "trial build result is incomplete");

  gtsam::Values disconnected_values = values;
  disconnected_values.insert(X(2), gtsam::Pose3());
  expect_throw<std::invalid_argument>([&] { glim::build_trial_isam2(factors, disconnected_values, gtsam::ISAM2Params()); }, "disconnected pose values must fail trial build");

  gtsam::Values isolated_auxiliary = values;
  isolated_auxiliary.insert(V(0), gtsam::Vector3(0.0, 0.0, 0.0));
  expect_throw([&] { glim::build_trial_isam2(factors, isolated_auxiliary, gtsam::ISAM2Params()); }, "unconstrained auxiliary values must fail trial build");

  gtsam::NonlinearFactorGraph underconstrained;
  underconstrained.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), gtsam::Pose3(), noise);
  underconstrained.emplace_shared<gtsam::RangeFactor<gtsam::Pose3, gtsam::Pose3>>(X(0), X(1), 1.0, gtsam::noiseModel::Isotropic::Sigma(1, 0.1));
  expect_throw(
    [&] { glim::build_trial_isam2(underconstrained, values, gtsam::ISAM2Params()); },
    "topologically connected but numerically underconstrained graph must fail trial build");
}

void test_session_transform() {
  gtsam::Values values;
  values.insert(X(3), gtsam::Pose3(gtsam::Rot3::Rz(0.2), gtsam::Point3(1.0, 0.0, 0.0)));
  values.insert(E(6), gtsam::Pose3(gtsam::Rot3::Rz(-0.1), gtsam::Point3(0.0, 2.0, 0.0)));
  values.insert(V(6), gtsam::Vector3(1.0, 2.0, 3.0));
  values.insert(B(6), gtsam::imuBias::ConstantBias(gtsam::Vector3(0.1, 0.2, 0.3), gtsam::Vector3(0.4, 0.5, 0.6)));

  const gtsam::Pose3 transform(gtsam::Rot3::Rz(1.5707963267948966), gtsam::Point3(10.0, 20.0, 30.0));
  const auto transformed = glim::transform_session_values(values, transform);

  expect(transformed.at<gtsam::Pose3>(X(3)).equals(transform * values.at<gtsam::Pose3>(X(3))), "X pose transform is incorrect");
  expect(transformed.at<gtsam::Pose3>(E(6)).equals(transform * values.at<gtsam::Pose3>(E(6))), "E pose transform is incorrect");
  expect(transformed.at<gtsam::Vector3>(V(6)).isApprox(transform.rotation().rotate(values.at<gtsam::Vector3>(V(6)))), "V rotation is incorrect");
  expect(transformed.at<gtsam::imuBias::ConstantBias>(B(6)).equals(values.at<gtsam::imuBias::ConstantBias>(B(6))), "bias must remain unchanged");
}

void test_merge_candidate() {
  const auto hard_noise = gtsam::noiseModel::Isotropic::Sigma(6, 0.1);
  gtsam::Values target_values;
  target_values.insert(X(0), gtsam::Pose3());
  target_values.insert(X(1), gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(10.0, 0.0, 0.0)));
  target_values.insert(X(2), gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(20.0, 0.0, 0.0)));

  gtsam::NonlinearFactorGraph target_factors;
  target_factors.emplace_shared<gtsam_points::LinearDampingFactor>(X(0), 6, 1e6);
  target_factors.emplace_shared<gtsam_points::LinearDampingFactor>(X(2), 6, 1e3);
  target_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(0), X(1), target_values.at<gtsam::Pose3>(X(0)).between(target_values.at<gtsam::Pose3>(X(1))), hard_noise);
  target_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(1), X(2), target_values.at<gtsam::Pose3>(X(1)).between(target_values.at<gtsam::Pose3>(X(2))), hard_noise);

  gtsam::Values source_values;
  source_values.insert(X(3), gtsam::Pose3());
  source_values.insert(X(4), gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0)));
  gtsam::NonlinearFactorGraph source_factors;
  source_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(3), X(4), gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0)), hard_noise);

  glim::SessionMergeOptions options;
  options.merge_factor = gtsam::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(0), X(3), gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(5.0, 0.0, 0.0)), hard_noise);
  options.prune_ranges = {{1, 1}};
  options.ensure_connected_graph = false;

  const auto disconnected = glim::build_session_merge_candidate(target_factors, target_values, source_factors, source_values, 3, 5, options);
  expect(!disconnected.values.exists(X(1)), "pruned pose remains in candidate values");
  expect(disconnected.values.at<gtsam::Pose3>(X(3)).translation().isApprox(gtsam::Point3(5.0, 0.0, 0.0)), "source session global transform is incorrect");
  expect(disconnected.factors.size() == 4, "disconnected candidate factor count is incorrect");
  expect(!disconnected.connectivity.all_poses_reachable(), "candidate must remain disconnected when soft bridges are disabled");
  expect(disconnected.connectivity.pose_component_count == 2, "disconnected candidate pose component count is incorrect");

  auto edited = disconnected;
  gtsam::NonlinearFactorGraph first_manual_factors;
  first_manual_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(3), X(4), gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0)), hard_noise);
  glim::append_candidate_factors(edited, first_manual_factors);
  expect(!edited.connectivity.all_poses_reachable(), "a manual factor within one component must not hide a disconnected component");

  gtsam::NonlinearFactorGraph connecting_manual_factor;
  connecting_manual_factor
    .emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(4), X(2), edited.values.at<gtsam::Pose3>(X(4)).between(edited.values.at<gtsam::Pose3>(X(2))), hard_noise);
  glim::append_candidate_factors(edited, connecting_manual_factor);
  expect(edited.connectivity.all_poses_reachable(), "manual factors failed to repair a disconnected candidate");
  expect(edited.diagnostic.empty(), "connected candidate retained a disconnected diagnostic");
  const auto edited_trial = glim::build_trial_isam2(edited.factors, edited.values, gtsam::ISAM2Params());
  expect(edited_trial.estimate.size() == edited.values.size(), "manually repaired candidate failed trial build");

  options.ensure_connected_graph = true;
  const auto connected = glim::build_session_merge_candidate(target_factors, target_values, source_factors, source_values, 3, 5, options);
  expect(connected.connectivity.all_poses_reachable(), "soft bridges failed to connect all candidate poses");
  expect(connected.factors.size() == 5, "soft bridging must add exactly one factor per disconnected target component");
  expect(connected.bridges.size() == 1, "soft bridge information is missing");
  expect(connected.bridges[0].from_id == 4 && connected.bridges[0].to_id == 2, "soft bridge information contains incorrect submap IDs");
  expect(std::abs(connected.bridges[0].distance - 14.0) < 1e-9, "soft bridge information contains an incorrect distance");
  const auto bridge_factor = dynamic_cast<const gtsam::BetweenFactor<gtsam::Pose3>*>(connected.factors.back().get());
  expect(bridge_factor && bridge_factor->key1() == X(4) && bridge_factor->key2() == X(2), "soft bridging did not select the nearest source-to-target pair");
  const auto trial_build = glim::build_trial_isam2(connected.factors, connected.values, gtsam::ISAM2Params());
  expect(trial_build.estimate.size() == connected.values.size(), "connected session merge candidate failed trial build");

  options.prune_ranges.clear();
  const auto ordinary_merge = glim::build_session_merge_candidate(target_factors, target_values, source_factors, source_values, 3, 5, options);
  expect(ordinary_merge.connectivity.all_poses_reachable(), "ordinary merge without pruning must remain connected");
  expect(ordinary_merge.bridges.empty(), "ordinary connected merge recorded an unexpected soft bridge");
  expect(ordinary_merge.values.size() == target_values.size() + source_values.size(), "ordinary merge lost values");
  expect(ordinary_merge.factors.size() == target_factors.size() + source_factors.size() + 1, "ordinary merge added unexpected factors");

  gtsam::Values target_with_historical_prune;
  target_with_historical_prune.insert(X(0), target_values.at<gtsam::Pose3>(X(0)));
  target_with_historical_prune.insert(X(2), target_values.at<gtsam::Pose3>(X(2)));
  gtsam::NonlinearFactorGraph target_with_historical_prune_factors;
  target_with_historical_prune_factors.emplace_shared<gtsam_points::LinearDampingFactor>(X(0), 6, 1e6);
  target_with_historical_prune_factors
    .emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(0), X(2), target_values.at<gtsam::Pose3>(X(0)).between(target_values.at<gtsam::Pose3>(X(2))), hard_noise);
  gtsam::Values source_with_historical_prune;
  source_with_historical_prune.insert(X(3), source_values.at<gtsam::Pose3>(X(3)));
  options.prune_ranges = {{1, 1}};
  const auto historical = glim::build_session_merge_candidate(target_with_historical_prune_factors, target_with_historical_prune, {}, source_with_historical_prune, 3, 5, options);
  expect(historical.requested_prune_ranges == std::vector<glim::SubmapRange>({{1, 1}}), "requested historical prune range was not retained");
  expect(historical.applied_prune_ranges.empty(), "an already-pruned target must not be applied again");
  expect(!historical.values.exists(X(1)) && !historical.values.exists(X(4)), "historically pruned poses were restored into the candidate");
  expect(historical.connectivity.all_poses_reachable(), "historically pruned source or target broke candidate connectivity");

  options.prune_ranges.clear();
  options.merge_factor = gtsam::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(3), X(0), gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(-5.0, 0.0, 0.0)), hard_noise);
  const auto reversed_merge = glim::build_session_merge_candidate(target_factors, target_values, source_factors, source_values, 3, 5, options);
  expect(reversed_merge.values.at<gtsam::Pose3>(X(3)).equals(ordinary_merge.values.at<gtsam::Pose3>(X(3))), "reversed merge factor changed the source transform");

  options.merge_factor = gtsam::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(0), X(3), gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(5.0, 0.0, 0.0)), hard_noise);
  options.prune_ranges = {{0, 0}};
  expect_throw<std::invalid_argument>(
    [&] { glim::build_session_merge_candidate(target_factors, target_values, source_factors, source_values, 3, 5, options); },
    "merge factor endpoint must not be pruned");
  options.prune_ranges = {{0, 2}};
  expect_throw<std::invalid_argument>(
    [&] { glim::build_session_merge_candidate(target_factors, target_values, source_factors, source_values, 3, 5, options); },
    "pruning every target submap must be rejected");
}

}  // namespace

int main() {
  try {
    test_submap_keys_and_filtering();
    test_filter_positions();
    test_pose_anchors();
    test_connectivity();
    test_trial_isam2_build();
    test_session_transform();
    test_merge_candidate();
  } catch (const std::exception& e) {
    std::cerr << "graph_edit_test failed: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
