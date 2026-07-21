import { createPinia, setActivePinia } from 'pinia';
import { describe, expect, it } from 'vitest';
import { useConfigStore } from '@/stores/config';

describe('config store normalization', () => {
  it('normalizes LSFG boolean strings before the capture settings render', () => {
    setActivePinia(createPinia());
    const store = useConfigStore();

    store.setConfig({
      lsfg_capture_framegen: 'enabled',
      lsfg_auto_flow_scale: 'false',
      lsfg_performance_mode: 'on',
      lsfg_adaptive_quality: '0',
    });

    expect(store.config.lsfg_capture_framegen).toBe(true);
    expect(store.config.lsfg_auto_flow_scale).toBe(false);
    expect(store.config.lsfg_performance_mode).toBe(true);
    expect(store.config.lsfg_adaptive_quality).toBe(false);
  });
});
