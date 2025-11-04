'use strict';

const fs = require('fs');
const path = require('path');

const { createGradient } = require('./gradient_composer');
const tokens = require('./lantern_tokens.json');

class GradientLineage {
  constructor(options = {}) {
    const baseDir = options.baseDir || __dirname;
    this.baseDir = baseDir;
    this.rootDir = options.rootDir || path.resolve(baseDir, '..', '..');
    this.tokens = options.tokens || (tokens && tokens.tokens ? tokens.tokens : {});
    this.aliases = this._buildAliases(options.aliases || {});

    const cssPaths = options.cssPaths || [path.join(baseDir, 'lantern_logo.css')];
    const svgPaths = options.svgPaths || [path.join(baseDir, 'lantern_logo.svg')];

    this.cssData = this._loadCssData(cssPaths);
    this.svgData = this._loadSvgData(svgPaths);
    this.tokenIndex = this._indexTokens(this.tokens);
    this.primitiveTokens = this._collectPrimitiveTokens(this.tokenIndex);
    this.hexToToken = this._buildHexIndex(this.tokenIndex, this.primitiveTokens);
  }

  trace(gradientIdentifier) {
    const gradientPath = this._normalizeGradientIdentifier(gradientIdentifier);
    const descriptor = this._getGradientDescriptor(gradientPath);

    if (!descriptor) {
      throw new RangeError(`Unknown gradient token: ${gradientIdentifier}`);
    }

    const gradientCss = this._composeGradientCss(descriptor);
    const primitives = this._collectPrimitives(descriptor);
    const compositionResult = this._collectCompositions(
      gradientPath,
      descriptor,
      gradientCss
    );
    const usage = this._collectUsage(compositionResult.aliases);

    return {
      gradient: gradientPath,
      css: gradientCss,
      primitives,
      compositions: compositionResult.entries,
      usage,
    };
  }

  _normalizeGradientIdentifier(identifier) {
    if (typeof identifier === 'string' && identifier.trim()) {
      const trimmed = identifier.trim();
      if (trimmed.startsWith('tokens.')) {
        return trimmed.slice('tokens.'.length);
      }
      return trimmed;
    }

    if (identifier && typeof identifier === 'object') {
      if (typeof identifier.path === 'string') {
        return this._normalizeGradientIdentifier(identifier.path);
      }
      if (typeof identifier.name === 'string') {
        return this._normalizeGradientIdentifier(identifier.name);
      }
    }

    throw new TypeError('Gradient identifier must be a non-empty string or object with path/name.');
  }

  _getGradientDescriptor(pathName) {
    if (!this.tokenIndex.has(pathName)) {
      return null;
    }
    const value = this.tokenIndex.get(pathName);
    if (value && typeof value === 'object' && !Array.isArray(value) && value.stops) {
      return value;
    }
    return null;
  }

  _composeGradientCss(descriptor) {
    if (!descriptor || !descriptor.stops) {
      return '';
    }
    const options = {};
    if (typeof descriptor.rotation === 'number') {
      options.rotation = descriptor.rotation;
    }
    return createGradient(descriptor.stops, options);
  }

  _collectPrimitives(descriptor) {
    const stops = Array.isArray(descriptor && descriptor.stops)
      ? descriptor.stops
      : [];
    const entries = new Map();

    stops.forEach((stop) => {
      if (!stop) {
        return;
      }
      const resolved = this._resolvePrimitive(stop.color);
      if (!resolved) {
        return;
      }
      const { token, value } = resolved;
      if (token && this.primitiveTokens.has(token) && !entries.has(token)) {
        const position = typeof stop.position === 'number' ? stop.position : null;
        entries.set(token, {
          token,
          value,
          position,
        });
      }
    });

    return Array.from(entries.values()).sort((a, b) => {
      if (a.position == null && b.position == null) {
        return a.token.localeCompare(b.token);
      }
      if (a.position == null) {
        return 1;
      }
      if (b.position == null) {
        return -1;
      }
      if (a.position === b.position) {
        return a.token.localeCompare(b.token);
      }
      return a.position - b.position;
    });
  }

  _resolvePrimitive(color) {
    if (typeof color !== 'string') {
      return null;
    }
    const trimmed = color.trim();
    if (!trimmed) {
      return null;
    }
    let reference = trimmed;
    if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
      reference = trimmed.slice(1, -1);
    }

    if (this.tokenIndex.has(reference) && this.primitiveTokens.has(reference)) {
      return { token: reference, value: this.tokenIndex.get(reference) };
    }

    if (/^#[0-9a-f]{6}$/i.test(reference)) {
      const normalized = reference.toLowerCase();
      if (this.hexToToken.has(normalized)) {
        const token = this.hexToToken.get(normalized);
        return { token, value: this.tokenIndex.get(token) };
      }
      return { token: null, value: normalized };
    }

    return null;
  }

  _collectCompositions(gradientPath, descriptor, gradientCss) {
    const entries = [
      {
        type: 'token',
        identifier: gradientPath,
        value: descriptor,
        css: gradientCss,
      },
    ];

    const aliases = new Set(this.aliases[gradientPath] || []);
    const normalizedGradient = this._normalizeCssValue(gradientCss);

    if (normalizedGradient) {
      this.cssData.customProperties.forEach((prop) => {
        if (this._normalizeCssValue(prop.value) === normalizedGradient) {
          aliases.add(prop.name);
          entries.push({
            type: 'css-custom-property',
            identifier: prop.name,
            file: prop.file,
            line: prop.line,
            value: prop.value,
          });
        }
      });
    }

    const gradientIds = new Set();
    aliases.forEach((alias) => {
      if (alias.startsWith('#')) {
        gradientIds.add(alias.slice(1));
      }
    });

    this.svgData.gradients.forEach((gradient) => {
      if (
        gradientIds.has(gradient.id) ||
        (gradientIds.size === 0 && this._svgGradientMatches(descriptor, gradient))
      ) {
        aliases.add(`#${gradient.id}`);
        entries.push({
          type: 'svg-gradient',
          identifier: `#${gradient.id}`,
          file: gradient.file,
          line: gradient.line,
          stops: gradient.stops,
        });
      }
    });

    return { entries, aliases };
  }

  _svgGradientMatches(descriptor, svgGradient) {
    if (!descriptor || !descriptor.stops || !svgGradient || !svgGradient.stops) {
      return false;
    }
    const tokenStops = descriptor.stops.map((stop) => {
      const resolved = this._resolvePrimitive(stop && stop.color);
      return resolved ? resolved.value : null;
    });

    if (tokenStops.length !== svgGradient.stops.length) {
      return false;
    }

    for (let i = 0; i < tokenStops.length; i += 1) {
      const expected = tokenStops[i];
      const actual = svgGradient.stops[i] && svgGradient.stops[i].fallback;
      if (!expected || !actual) {
        return false;
      }
      if (expected.toLowerCase() !== actual.toLowerCase()) {
        return false;
      }
    }

    return true;
  }

  _collectUsage(aliases) {
    const usage = [];
    const varNames = new Set();
    const gradientIds = new Set();

    aliases.forEach((alias) => {
      if (alias.startsWith('--')) {
        varNames.add(alias);
      } else if (alias.startsWith('#')) {
        gradientIds.add(alias.slice(1));
      }
    });

    this.cssData.varUsages.forEach((entry) => {
      if (varNames.has(entry.name)) {
        usage.push({
          type: 'css-var',
          identifier: entry.name,
          file: entry.file,
          line: entry.line,
          context: entry.context,
        });
      }
    });

    this.cssData.urlReferences.forEach((entry) => {
      if (gradientIds.has(entry.id)) {
        usage.push({
          type: 'css-url',
          identifier: `#${entry.id}`,
          file: entry.file,
          line: entry.line,
          context: entry.context,
        });
      }
    });

    return usage;
  }

  _buildAliases(optionAliases) {
    const defaults = {
      'gradient.brand.primary': ['--brand-gradient', '#lantern-gradient'],
    };
    const aliases = {};
    Object.keys(defaults).forEach((key) => {
      aliases[key] = defaults[key].slice();
    });
    Object.keys(optionAliases).forEach((key) => {
      if (!aliases[key]) {
        aliases[key] = [];
      }
      const values = Array.isArray(optionAliases[key])
        ? optionAliases[key]
        : [optionAliases[key]];
      values.forEach((value) => {
        if (typeof value === 'string' && value.trim()) {
          aliases[key].push(value.trim());
        }
      });
    });
    return aliases;
  }

  _loadCssData(paths) {
    const customProperties = [];
    const varUsages = [];
    const urlReferences = [];

    paths.forEach((cssPath) => {
      const absolute = path.isAbsolute(cssPath)
        ? cssPath
        : path.join(this.baseDir, cssPath);
      if (!fs.existsSync(absolute)) {
        return;
      }
      const content = fs.readFileSync(absolute, 'utf8');
      const relative = path.relative(this.rootDir, absolute);

      const propertyRegex = /(--[a-z0-9-]+)\s*:\s*([^;]+);/gims;
      let propertyMatch;
      while ((propertyMatch = propertyRegex.exec(content)) !== null) {
        const [, name, value] = propertyMatch;
        const line = this._lineNumberAtIndex(content, propertyMatch.index);
        customProperties.push({
          name,
          value: value.trim(),
          file: relative,
          line,
        });
      }

      const varRegex = /var\(\s*(--[a-z0-9-]+)\s*(?:,[^)]+)?\)/gim;
      let varMatch;
      while ((varMatch = varRegex.exec(content)) !== null) {
        const [, name] = varMatch;
        const line = this._lineNumberAtIndex(content, varMatch.index);
        const context = this._lineAtIndex(content, varMatch.index);
        varUsages.push({
          name,
          file: relative,
          line,
          context,
        });
      }

      const urlRegex = /url\(\s*#([^)]+)\s*\)/gim;
      let urlMatch;
      while ((urlMatch = urlRegex.exec(content)) !== null) {
        const [, id] = urlMatch;
        const line = this._lineNumberAtIndex(content, urlMatch.index);
        const context = this._lineAtIndex(content, urlMatch.index);
        urlReferences.push({
          id,
          file: relative,
          line,
          context,
        });
      }
    });

    return { customProperties, varUsages, urlReferences };
  }

  _loadSvgData(paths) {
    const gradients = [];

    paths.forEach((svgPath) => {
      const absolute = path.isAbsolute(svgPath)
        ? svgPath
        : path.join(this.baseDir, svgPath);
      if (!fs.existsSync(absolute)) {
        return;
      }
      const content = fs.readFileSync(absolute, 'utf8');
      const relative = path.relative(this.rootDir, absolute);

      const gradientRegex = /<linearGradient[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/linearGradient>/gim;
      let gradientMatch;
      while ((gradientMatch = gradientRegex.exec(content)) !== null) {
        const [, id, body] = gradientMatch;
        const line = this._lineNumberAtIndex(content, gradientMatch.index);
        const stops = [];
        const stopRegex = /<stop[^>]*offset="([^"]+)"[^>]*stop-color="([^"]+)"[^>]*\/>/gim;
        let stopMatch;
        while ((stopMatch = stopRegex.exec(body)) !== null) {
          const [, offset, color] = stopMatch;
          stops.push({
            offset,
            color,
            fallback: this._extractHex(color),
          });
        }
        gradients.push({
          id,
          file: relative,
          line,
          stops,
        });
      }
    });

    return { gradients };
  }

  _indexTokens(node, prefix = []) {
    const index = new Map();

    const walk = (current, currentPath) => {
      if (!current || typeof current !== 'object') {
        return;
      }
      Object.keys(current).forEach((key) => {
        const value = current[key];
        const nextPath = currentPath.concat(key);
        if (value && typeof value === 'object' && !Array.isArray(value) && 'value' in value) {
          index.set(nextPath.join('.'), value.value);
          if (value.value && typeof value.value === 'object' && !Array.isArray(value.value)) {
            walk(value.value, nextPath);
          }
        } else if (value && typeof value === 'object') {
          walk(value, nextPath);
        }
      });
    };

    walk(node, prefix);
    return index;
  }

  _collectPrimitiveTokens(index) {
    const primitives = new Set();
    index.forEach((value, key) => {
      if (typeof value === 'string' && key.startsWith('color.')) {
        primitives.add(key);
      }
    });
    return primitives;
  }

  _buildHexIndex(index, primitives) {
    const hexIndex = new Map();
    index.forEach((value, key) => {
      if (typeof value === 'string' && primitives.has(key)) {
        hexIndex.set(value.toLowerCase(), key);
      }
    });
    return hexIndex;
  }

  _normalizeCssValue(value) {
    if (typeof value !== 'string') {
      return '';
    }
    return value.replace(/\s+/g, '').toLowerCase();
  }

  _lineNumberAtIndex(content, index) {
    return content.slice(0, index).split(/\r?\n/).length;
  }

  _lineAtIndex(content, index) {
    const start = content.lastIndexOf('\n', index);
    const end = content.indexOf('\n', index);
    const lineStart = start === -1 ? 0 : start + 1;
    const lineEnd = end === -1 ? content.length : end;
    return content.slice(lineStart, lineEnd).trim();
  }

  _extractHex(value) {
    if (typeof value !== 'string') {
      return '';
    }
    const match = value.match(/#([0-9a-f]{3,8})/i);
    return match ? `#${match[1]}` : '';
  }
}

module.exports = GradientLineage;
