'use strict';

const tokens = require('./lantern_tokens.json');

const BRAND_COLOR_KEY = ['tokens', 'color', 'brand'];

const brandPalette = BRAND_COLOR_KEY.reduce(
  (acc, key) => (acc && acc[key] ? acc[key] : {}),
  tokens
);

const brandColors = Object.entries(brandPalette);

const PRIMITIVE_REF_SET = new Set(
  brandColors.map(([name]) => `color.brand.${name}`)
);

const PRIMITIVE_HEX_SET = new Set(
  brandColors
    .map(([, descriptor]) => descriptor.value)
    .filter((value) => typeof value === 'string')
    .map((value) => value.trim().toLowerCase())
);

function normalizeTokenReference(color) {
  if (typeof color !== 'string') {
    return '';
  }
  const trimmed = color.trim();
  if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function isPrimitive(color) {
  const ref = normalizeTokenReference(color);

  if (PRIMITIVE_REF_SET.has(ref)) {
    return true;
  }

  if (/^#[0-9a-f]{6}$/i.test(ref)) {
    return PRIMITIVE_HEX_SET.has(ref.toLowerCase());
  }

  return false;
}

function toCssCustomProperty(ref) {
  if (ref.startsWith('color.brand.')) {
    const [, , ...parts] = ref.split('.');
    return `var(--brand-${parts.join('-')})`;
  }
  return ref;
}

function resolveColor(color) {
  const ref = normalizeTokenReference(color);
  if (PRIMITIVE_REF_SET.has(ref)) {
    return toCssCustomProperty(ref);
  }
  return color;
}

function formatStopPosition(position) {
  if (typeof position !== 'number' || Number.isNaN(position)) {
    return position;
  }
  const percentage = position * 100;
  return `${Number.parseFloat(percentage.toFixed(2))}%`;
}

function createGradient(stops, options = {}) {
  const { rotation = 160 } = options;
  const orderedStops = [...stops].sort((a, b) => {
    if (typeof a.position !== 'number' || typeof b.position !== 'number') {
      return 0;
    }
    return a.position - b.position;
  });

  const stopList = orderedStops
    .map((stop) => {
      const color = resolveColor(stop.color);
      const position = formatStopPosition(stop.position);
      return position ? `${color} ${position}` : color;
    })
    .join(', ');

  return `linear-gradient(${rotation}deg, ${stopList})`;
}

function composeGradient(stops, options = {}) {
  const { allowOrphans = false } = options;

  if (!allowOrphans) {
    stops.forEach((stop) => {
      if (!isPrimitive(stop.color)) {
        console.warn(
          `Orphan color ${stop.color} detected. ` +
            'Consider adding to primitives or set allowOrphans: true'
        );
      }
    });
  }

  return createGradient(stops, options);
}

module.exports = {
  composeGradient,
  createGradient,
  isPrimitive,
  resolveColor,
};
