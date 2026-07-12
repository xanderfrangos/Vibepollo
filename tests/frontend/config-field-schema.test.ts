import { getConfigFieldDefinition } from '@web/configs/configFieldSchema';

const baseContext = {
  t: (key: string) => key,
  platform: 'windows',
  metadata: {},
};

describe('configFieldSchema', () => {
  test.each([0, 1, '0', '1'])(
    'keeps back_button_timeout as a number field for %p',
    (currentValue) => {
      expect(
        getConfigFieldDefinition('back_button_timeout', {
          ...baseContext,
          defaultValue: -1,
          currentValue,
        }).kind,
      ).toBe('number');
    },
  );

  test('anchors known fields to the default type instead of the live edited value', () => {
    expect(
      getConfigFieldDefinition('system_tray', {
        ...baseContext,
        defaultValue: true,
        currentValue: '0',
      }).kind,
    ).toBe('checkbox');

    expect(
      getConfigFieldDefinition('remember_me_refresh_token_ttl_seconds', {
        ...baseContext,
        defaultValue: 604800,
        currentValue: 'enabled',
      }).kind,
    ).toBe('number');
  });

  test('falls back to the current value when no default is available', () => {
    expect(
      getConfigFieldDefinition('unknown_number_key', {
        ...baseContext,
        currentValue: 1,
      }).kind,
    ).toBe('number');

    expect(
      getConfigFieldDefinition('unknown_bool_key', {
        ...baseContext,
        currentValue: 'enabled',
      }).kind,
    ).toBe('checkbox');
  });

  test('renders adaptive LSFG quality as a switch', () => {
    expect(
      getConfigFieldDefinition('lsfg_adaptive_quality', {
        ...baseContext,
        defaultValue: true,
        currentValue: true,
      }).kind,
    ).toBe('switch');
  });

  test('renders automatic LSFG flow scale as a switch', () => {
    expect(
      getConfigFieldDefinition('lsfg_auto_flow_scale', {
        ...baseContext,
        defaultValue: true,
        currentValue: true,
      }).kind,
    ).toBe('switch');
  });

  test('constrains LSFG pacing grace to its safe range', () => {
    expect(
      getConfigFieldDefinition('lsfg_pacing_grace_ms', {
        ...baseContext,
        currentValue: 0,
        defaultValue: 0,
      }),
    ).toMatchObject({ kind: 'slider', min: 0, max: 6, step: 1, precision: 0 });
  });

  test('renders LSFG maximum multiplier as a 2-10 slider', () => {
    expect(
      getConfigFieldDefinition('lsfg_max_multiplier', {
        ...baseContext,
        currentValue: 4,
        defaultValue: 4,
      }),
    ).toMatchObject({ kind: 'slider', min: 2, max: 10, step: 1, precision: 0 });
  });
});
