export default {
  rules: {
    "no-invalid-position-at-import-rule": true,
    "selector-attribute-operator-disallowed-list": ["*="],
    "declaration-property-value-disallowed-list": {
      transition: ["/\\ball\\b/"],
      "transition-property": ["/\\ball\\b/"]
    }
  }
};
