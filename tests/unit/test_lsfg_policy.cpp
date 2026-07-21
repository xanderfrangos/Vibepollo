/**
 * @file tests/unit/test_lsfg_policy.cpp
 * @brief Pure policy tests for host-side LSFG configuration.
 */

#ifdef _WIN32

  #include <gtest/gtest.h>

  #include "src/platform/windows/lsfg_framegen.h"

namespace {

  using platf::dxgi::lsfg_framegen_t;
  using namespace std::chrono_literals;

  TEST(LsfgPolicy, AutomaticFlowScaleTracksProcessedTextureSize) {
    EXPECT_EQ(lsfg_framegen_t::automatic_flow_scale_percent(1280, 720), 100);
    EXPECT_EQ(lsfg_framegen_t::automatic_flow_scale_percent(1920, 1080), 100);
    EXPECT_GT(lsfg_framegen_t::automatic_flow_scale_percent(2560, 1440), 50);
    EXPECT_LT(lsfg_framegen_t::automatic_flow_scale_percent(2560, 1440), 100);
    EXPECT_EQ(lsfg_framegen_t::automatic_flow_scale_percent(3840, 2160), 50);
    EXPECT_EQ(lsfg_framegen_t::automatic_flow_scale_percent(7680, 4320), 50);
  }

  TEST(LsfgPolicy, AdaptiveOffBuildsOnlyConfiguredProfile) {
    lsfg_framegen_t::options_t base;
    base.flow_scale = 1.0f;

    const auto profiles = lsfg_framegen_t::quality_profiles(base, false);

    ASSERT_EQ(profiles.size(), 1u);
    EXPECT_FLOAT_EQ(profiles.front().flow_scale, 1.0f);
    EXPECT_FALSE(profiles.front().performance_mode);
  }

  TEST(LsfgPolicy, AdaptiveProfilesAreDeduplicatedAtMinimumScale) {
    lsfg_framegen_t::options_t base;
    base.flow_scale = 0.25f;
    base.performance_mode = true;

    const auto profiles = lsfg_framegen_t::quality_profiles(base, true);

    ASSERT_EQ(profiles.size(), 1u);
    EXPECT_FLOAT_EQ(profiles.front().flow_scale, 0.25f);
    EXPECT_TRUE(profiles.front().performance_mode);
  }

  TEST(LsfgPolicy, AdaptiveProfilesDescendWithoutDuplicates) {
    lsfg_framegen_t::options_t base;
    base.flow_scale = 0.5f;
    base.performance_mode = false;

    const auto profiles = lsfg_framegen_t::quality_profiles(base, true);

    ASSERT_EQ(profiles.size(), 3u);
    EXPECT_FLOAT_EQ(profiles[0].flow_scale, 0.5f);
    EXPECT_FALSE(profiles[0].performance_mode);
    EXPECT_FLOAT_EQ(profiles[1].flow_scale, 0.25f);
    EXPECT_FALSE(profiles[1].performance_mode);
    EXPECT_FLOAT_EQ(profiles[2].flow_scale, 0.25f);
    EXPECT_TRUE(profiles[2].performance_mode);
  }

  TEST(LsfgTimeline, GeneratesMidpointForThirtyToSixtyConversion) {
    const std::chrono::steady_clock::time_point previous {100s};
    const auto source_interval = 33333333ns;
    const auto target_interval = 16666667ns;
    const auto current = previous + source_interval;
    auto timeline = lsfg_framegen_t::create_test_timeline(
      previous,
      current,
      source_interval,
      target_interval
    );

    float phase = 0.0f;
    EXPECT_FALSE(timeline->want_generated(current, phase));
    timeline->mark_passthrough_shown();
    EXPECT_TRUE(timeline->want_generated(current + target_interval, phase));
    EXPECT_NEAR(phase, 0.5f, 0.01f);
    timeline->mark_generated_shown();
    EXPECT_FALSE(timeline->want_generated(current + source_interval, phase));
  }

  TEST(LsfgTimeline, SupportsFractionalThirtyThreeToSixtyConversion) {
    const std::chrono::steady_clock::time_point previous {100s};
    const auto source_interval = 30303030ns;
    const auto target_interval = 16666667ns;
    const auto current = previous + source_interval;
    auto timeline = lsfg_framegen_t::create_test_timeline(
      previous,
      current,
      source_interval,
      target_interval
    );

    float phase = 0.0f;
    EXPECT_FALSE(timeline->want_generated(current, phase));
    timeline->mark_passthrough_shown();
    EXPECT_TRUE(timeline->want_generated(current + target_interval, phase));
    EXPECT_GT(phase, 0.5f);
    EXPECT_LT(phase, 0.6f);
  }

  TEST(LsfgTimeline, GeneratesMidpointForSixtyToOneTwentyConversion) {
    const std::chrono::steady_clock::time_point previous {100s};
    const auto source_interval = 16666667ns;
    const auto target_interval = 8333333ns;
    const auto current = previous + source_interval;
    auto timeline = lsfg_framegen_t::create_test_timeline(
      previous,
      current,
      source_interval,
      target_interval
    );

    float phase = 0.0f;
    EXPECT_FALSE(timeline->want_generated(current, phase));
    timeline->mark_passthrough_shown();
    EXPECT_TRUE(timeline->want_generated(current + target_interval, phase));
    EXPECT_NEAR(phase, 0.5f, 0.01f);
  }

  TEST(LsfgTimeline, StopsGeneratingAtConfiguredMultiplierCap) {
    const std::chrono::steady_clock::time_point previous {100s};
    const auto source_interval = 100ms;
    const auto target_interval = 16ms;
    const auto current = previous + source_interval;
    auto timeline = lsfg_framegen_t::create_test_timeline(
      previous,
      current,
      source_interval,
      target_interval,
      4
    );

    float phase = 0.0f;
    EXPECT_TRUE(timeline->want_generated(current, phase));
    EXPECT_NEAR(phase, 0.25f, 0.001f);
    timeline->mark_generated_shown();
    EXPECT_FALSE(timeline->want_generated(current + 3 * target_interval, phase));
  }

}  // namespace

#endif
