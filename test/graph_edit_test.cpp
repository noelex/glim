#include <glim/mapping/graph_edit.hpp>

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/sam/RangeFactor.h>
#include <gtsam/slam/BetweenFactor.h>

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
  values.insert(keys.pose, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(submap_id, 0.0, 0.0)));
  values.insert(keys.extrinsics[0], gtsam::Pose3());
  values.insert(keys.extrinsics[1], gtsam::Pose3());
  values.insert(keys.velocities[0], gtsam::Vector3(0.0, 0.0, 0.0));
  values.insert(keys.velocities[1], gtsam::Vector3(0.0, 0.0, 0.0));
  values.insert(keys.biases[0], gtsam::imuBias::ConstantBias());
  values.insert(keys.biases[1], gtsam::imuBias::ConstantBias());
}

void test_submap_keys_and_filtering() {
  const auto keys = glim::submap_state_keys(3);
  expect(keys.pose == X(3), "submap pose key is incorrect");
  expect(keys.extrinsics == std::array<gtsam::Key, 2>{E(6), E(7)}, "submap extrinsic keys are incorrect");
  expect(keys.velocities == std::array<gtsam::Key, 2>{V(6), V(7)}, "submap velocity keys are incorrect");
  expect(keys.biases == std::array<gtsam::Key, 2>{B(6), B(7)}, "submap bias keys are incorrect");
  expect(glim::submap_state_key_set({1, 1, 2}).size() == 14, "submap key set must remove duplicate IDs");
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

  const auto removed_keys = glim::submap_state_key_set({1});
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
  original_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), original_prior, anchor_noise);
  original_factors.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(0), gtsam::Vector3::Zero(), gtsam::noiseModel::Isotropic::Sigma(3, 1.0));

  const auto original_anchor = glim::find_unique_full_pose_anchor(original_factors);
  expect(original_anchor.key == X(0), "full pose anchor key is incorrect");
  expect(original_anchor.prior.equals(original_prior), "full pose anchor prior is incorrect");

  gtsam::Values candidate_values;
  const gtsam::Pose3 target_pose(gtsam::Rot3::Rz(0.4), gtsam::Point3(5.0, 6.0, 7.0));
  candidate_values.insert(X(1), target_pose);
  candidate_values.insert(X(2), gtsam::Pose3());
  gtsam::NonlinearFactorGraph candidate_factors;
  glim::ensure_pose_anchor(candidate_factors, candidate_values, original_anchor, {2, 1});

  const auto transferred = glim::find_unique_full_pose_anchor(candidate_factors);
  expect(transferred.key == X(1), "anchor must transfer to the lowest active target submap ID");
  expect(transferred.prior.equals(target_pose), "transferred anchor must use the current global pose");
  expect(transferred.noise_model->equals(*anchor_noise), "transferred anchor must preserve the original noise model");

  expect_throw<std::invalid_argument>([] { glim::find_unique_full_pose_anchor({}); }, "zero full pose anchors must be rejected");

  gtsam::NonlinearFactorGraph multiple_anchors;
  multiple_anchors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), gtsam::Pose3(), anchor_noise);
  multiple_anchors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(1), gtsam::Pose3(), anchor_noise);
  expect_throw<std::invalid_argument>([&] { glim::find_unique_full_pose_anchor(multiple_anchors); }, "multiple full pose anchors must be rejected");

  gtsam::Values old_anchor_still_active;
  old_anchor_still_active.insert(X(0), gtsam::Pose3());
  gtsam::NonlinearFactorGraph missing_anchor;
  expect_throw<std::invalid_argument>(
    [&] { glim::ensure_pose_anchor(missing_anchor, old_anchor_still_active, original_anchor, {0}); },
    "a missing anchor must not be silently recreated when its original value remains active");
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

  const auto connectivity = glim::analyze_key_factor_connectivity(values, factors, X(0));
  expect(connectivity.components.size() == 3, "full key-factor component count is incorrect");
  expect(connectivity.pose_component_count == 2, "pose component count is incorrect");
  expect(connectivity.unreachable_pose_keys == gtsam::KeyVector{X(2)}, "unreachable pose detection is incorrect");

  gtsam::Values without_x2;
  for (const auto& value : values) {
    if (value.key != X(2)) {
      without_x2.insert(value.key, value.value);
    }
  }
  const auto isolated_auxiliary = glim::analyze_key_factor_connectivity(without_x2, factors, X(0));
  expect(isolated_auxiliary.all_poses_reachable(), "isolated E/V/B state must not fail pose connectivity");
  expect(isolated_auxiliary.components.size() == 2, "isolated auxiliary state must remain visible in full components");

  gtsam::NonlinearFactorGraph missing_key_factor;
  missing_key_factor.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(0), X(9), gtsam::Pose3(), noise);
  expect_throw<std::invalid_argument>(
    [&] { glim::analyze_key_factor_connectivity(values, missing_key_factor, X(0)); },
    "factor references to missing keys must be rejected");
}

void test_strict_trial_build() {
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(6, 0.1);
  gtsam::Values values;
  values.insert(X(0), gtsam::Pose3());
  values.insert(X(1), gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0)));

  gtsam::NonlinearFactorGraph factors;
  factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), gtsam::Pose3(), noise);
  factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(0), X(1), gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0)), noise);

  const auto result = glim::strict_trial_isam2_build(factors, values, gtsam::ISAM2Params());
  expect(result.isam2 != nullptr, "strict trial build did not return the temporary iSAM2 instance");
  expect(result.estimate.size() == values.size(), "strict trial build result is incomplete");

  gtsam::Values disconnected_values = values;
  disconnected_values.insert(X(2), gtsam::Pose3());
  expect_throw<std::invalid_argument>(
    [&] { glim::strict_trial_isam2_build(factors, disconnected_values, gtsam::ISAM2Params()); },
    "disconnected pose values must fail strict trial build");

  gtsam::Values isolated_auxiliary = values;
  isolated_auxiliary.insert(V(0), gtsam::Vector3(0.0, 0.0, 0.0));
  expect_throw(
    [&] { glim::strict_trial_isam2_build(factors, isolated_auxiliary, gtsam::ISAM2Params()); },
    "unconstrained auxiliary values must fail strict trial build");

  gtsam::NonlinearFactorGraph underconstrained;
  underconstrained.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), gtsam::Pose3(), noise);
  underconstrained.emplace_shared<gtsam::RangeFactor<gtsam::Pose3, gtsam::Pose3>>(X(0), X(1), 1.0, gtsam::noiseModel::Isotropic::Sigma(1, 0.1));
  expect_throw(
    [&] { glim::strict_trial_isam2_build(underconstrained, values, gtsam::ISAM2Params()); },
    "topologically connected but numerically underconstrained graph must fail strict trial build");
}

}  // namespace

int main() {
  try {
    test_submap_keys_and_filtering();
    test_filter_positions();
    test_pose_anchors();
    test_connectivity();
    test_strict_trial_build();
  } catch (const std::exception& e) {
    std::cerr << "graph_edit_test failed: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
