function buildRequiredElementError(method, selector) {
  return new Error(`Portal DOM contract missing required ${method}: ${selector}`);
}

export function shouldEnableDomAssertions(windowRef) {
  const host = String(windowRef?.location?.hostname || "").trim().toLowerCase();
  return host === "localhost" || host === "127.0.0.1" || host.endsWith(".local");
}

export function createDomContract(documentRef, { devAssertions = false } = {}) {
  function id(elementId, { required = false } = {}) {
    const node = documentRef.getElementById(String(elementId || ""));
    if (!node && required && devAssertions) {
      throw buildRequiredElementError("id", `#${elementId}`);
    }
    return node;
  }

  function query(selector, { required = false } = {}) {
    const node = documentRef.querySelector(String(selector || ""));
    if (!node && required && devAssertions) {
      throw buildRequiredElementError("selector", selector);
    }
    return node;
  }

  function assertPresent(elements, requiredKeys) {
    if (!devAssertions) return;
    const missingKeys = [];
    for (const key of requiredKeys) {
      if (!elements?.[key]) {
        missingKeys.push(String(key));
      }
    }
    if (missingKeys.length) {
      throw new Error(`Portal DOM contract missing required elements: ${missingKeys.join(", ")}`);
    }
  }

  return {
    id,
    query,
    assertPresent
  };
}
