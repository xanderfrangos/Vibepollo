/**
 * @file tests/unit/platform/windows/test_lsfg_timing.cpp
 */
#include "../../../tests_common.h"

#ifdef _WIN32
  #include "src/platform/windows/lsfg_framegen.h"

namespace {
  using namespace platf::dxgi::lsfg_timing;

  TEST(LsfgTiming, RejectsDuplicateAndBackwardQpcFrames) {
    EXPECT_TRUE(accept_source_qpc(0, 100));
    EXPECT_TRUE(accept_source_qpc(100, 0));
    EXPECT_TRUE(accept_source_qpc(100, 101));
    EXPECT_FALSE(accept_source_qpc(100, 100));
    EXPECT_FALSE(accept_source_qpc(100, 99));
  }

  TEST(LsfgTiming, GenerationHysteresisHandlesCommonConversions) {
    EXPECT_TRUE(update_generation_active(false, 60.0 / 30.0));
    EXPECT_TRUE(update_generation_active(false, 60.0 / 33.0));
    EXPECT_FALSE(update_generation_active(false, 60.0 / 59.0));

    EXPECT_TRUE(update_generation_active(true, 60.0 / 58.0));
    EXPECT_FALSE(update_generation_active(true, 60.0 / 59.0));
  }

  TEST(LsfgTiming, UsesExactClientRefreshRateWhenAvailable) {
    EXPECT_NEAR(target_fps(60, 5994), 60000.0 / 1001.0, 0.0001);
    EXPECT_NEAR(target_fps(120, 11988), 120000.0 / 1001.0, 0.0001);
    EXPECT_DOUBLE_EQ(target_fps(120, 0), 120.0);
  }

  TEST(LsfgTiming, AutomaticFlowScaleTracksCaptureResolution) {
    EXPECT_EQ(automatic_flow_scale_percent(1280, 720), 100);
    EXPECT_EQ(automatic_flow_scale_percent(1920, 1080), 100);
    EXPECT_EQ(automatic_flow_scale_percent(3840, 2160), 50);
    EXPECT_EQ(automatic_flow_scale_percent(7680, 4320), 50);
    EXPECT_GT(automatic_flow_scale_percent(2560, 1440), 50);
    EXPECT_LT(automatic_flow_scale_percent(2560, 1440), 100);
  }
}  // namespace
#endif
