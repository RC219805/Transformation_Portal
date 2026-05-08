import { escapeHtml, FRONTDOOR_ASSETS, renderBrandAsset } from "./brand.js";

const NAV_LINKS = Object.freeze([
  { id: "platform", label: "Platform" },
  { id: "workflow", label: "Workflow" },
  { id: "proof", label: "Proof" },
  { id: "teams", label: "Teams" },
  { id: "faq", label: "FAQ" }
]);

const HERO_SIGNALS = Object.freeze([
  "Offline verification report included",
  "Rights-aware approval gates",
  "Content Credentials-compatible export when enabled"
]);

const HERO_HIGHLIGHTS = Object.freeze([
  {
    title: "Reviewer-ready evidence",
    detail:
      "Provenance, rights posture, and review context stay attached from ingest through release."
  },
  {
    title: "Deterministic verification",
    detail:
      "Offline verification reports surface pass-fail status and Merkle root checks for independent inspection."
  },
  {
    title: "Partner-safe packaging",
    detail:
      "Approved outputs ship with manifests, review history, and operator decisions already assembled."
  }
]);

const HERO_PATHS = Object.freeze([
  {
    kicker: "Learn",
    title: "Explore the governed workflow",
    detail: "See how verify, enhance, enforce, and distribute fit together before you touch the console.",
    href: "#workflow",
    dataUi: "homepage-learn-link"
  },
  {
    kicker: "Verify",
    title: "Inspect the proof surface",
    detail: "Review the public verification report, bundle structure, and standards posture first.",
    href: "#proof-report",
    dataUi: "homepage-verify-link"
  },
  {
    kicker: "Operator Access",
    title: "Enter the managed console",
    detail: "Resume build, operate, and review work inside the governed operator shell.",
    href: "/login",
    dataUi: "homepage-access-link"
  }
]);

const OUTCOME_CARDS = Object.freeze([
  {
    title: "Approve faster",
    description:
      "Put provenance, rights posture, and review evidence in one place so counsel, brand teams, and distribution partners do not reconstruct context by hand."
  },
  {
    title: "Protect licensing posture",
    description:
      "Keep usage boundaries, disclosure notes, and operator decisions attached to the asset before it leaves the workflow."
  },
  {
    title: "Export proof, not just files",
    description:
      "Ship approved renditions alongside manifests, review history, and comparison surfaces reviewers can inspect quickly."
  }
]);

const EVIDENCE_ITEMS = Object.freeze([
  {
    title: "Verification report",
    description:
      "Machine-readable offline verifier output containing pass-fail status, counts, and root checks."
  },
  {
    title: "Provenance manifest",
    description:
      "Deterministic bindings for source bytes, metadata hashes, and origin entries."
  },
  {
    title: "Merkle root anchor",
    description:
      "A compact integrity anchor computed over the approved asset set."
  },
  {
    title: "Rights and review trail",
    description:
      "Operator approvals, comparison references, and policy decisions preserved with the handoff."
  }
]);

const WORKFLOW_STEPS = Object.freeze([
  {
    number: "01",
    title: "Verify",
    description:
      "Collect provenance inputs, source context, and ownership signals at ingest before the asset enters circulation.",
    output: "Capture metadata + file hash"
  },
  {
    number: "02",
    title: "Enhance",
    description:
      "Normalize quality, create delivery renditions, and preserve review context across finishing steps.",
    output: "Approved renditions + review context"
  },
  {
    number: "03",
    title: "Enforce",
    description:
      "Apply rights posture, disclosure rules, and internal release gates before anything is dispatched.",
    output: "Rights decisions + release gates"
  },
  {
    number: "04",
    title: "Distribute",
    description:
      "Export the approved asset alongside manifests, review records, and partner-ready packaging.",
    output: "Verification report + release bundle"
  }
]);

const AUDIENCE_SEGMENTS = Object.freeze([
  {
    title: "Creators",
    summary:
      "Package premium outputs with authorship, provenance context, and reuse posture from the start.",
    bullets: ["Protect authorship", "Monetize with cleaner handoffs"]
  },
  {
    title: "Studios",
    summary:
      "Maintain consistent evidence capture across complex editorial, finishing, and release pipelines.",
    bullets: ["Align post-production teams", "Simplify legal and partner review"]
  },
  {
    title: "Festivals and archives",
    summary:
      "Preserve chain of custody across intake, curation, exhibition, and long-term retention.",
    bullets: ["Keep history legible", "Standardize intake and handoff"]
  },
  {
    title: "Brands and distributors",
    summary:
      "Ship campaigns and catalog assets with rights clarity and disclosure readiness under scrutiny.",
    bullets: ["Reduce downstream friction", "Support regional policy needs"]
  }
]);

const PRINCIPLES = Object.freeze([
  {
    title: "Certification, not mythmaking",
    description:
      "DNA packages provenance context, rights posture, and workflow evidence. It does not claim to prove the truth of the depicted event."
  },
  {
    title: "Standards-compatible export",
    description:
      "Support Content Credentials-compatible manifests and machine-readable disclosure workflows when enabled and appropriate to the asset."
  },
  {
    title: "Real-world platform limits acknowledged",
    description:
      "Some downstream platforms strip metadata. DNA therefore pairs exportable credentials with internal review history and controlled release records."
  }
]);

const FAQ_ITEMS = Object.freeze([
  {
    question: "What can a reviewer actually inspect?",
    answer:
      "A machine-readable offline verification report, a provenance manifest with deterministic bindings, a Merkle root anchor, declared rights posture, and preserved operator review history."
  },
  {
    question: "Are signatures or timestamps always included?",
    answer:
      "No. Those are optional proof attachments when enabled. The core default proof surface is the verification report plus deterministic provenance artifacts."
  },
  {
    question: "Does DNA claim to prove that media is true?",
    answer:
      "No. The product is framed as a provenance and release-control layer, not a truth engine or a universal deepfake detector."
  },
  {
    question: "Will metadata survive every downstream platform?",
    answer:
      "No system should promise that. DNA can emit Content Credentials-compatible manifests when enabled, while also preserving workflow evidence for review when downstream services strip metadata."
  }
]);

const PREVIEW_ARTIFACTS = Object.freeze([
  {
    name: "verification_report.tp.meta.verification_report.v1.json",
    detail: "Offline verifier output with pass-fail status, counts, and root checks.",
    status: "Verified"
  },
  {
    name: "provenance_manifest.tp.meta.provenance.v1.json",
    detail: "Per-asset bindings for source bytes, metadata hashes, and provenance entries.",
    status: "Bound"
  },
  {
    name: "provenance_merkle.tp.meta.provenance_merkle.v1.json",
    detail: "Compact integrity anchor over the approved asset set.",
    status: "Computed"
  },
  {
    name: "review-log.ndjson",
    detail: "Operator approvals, comparisons, and release checkpoints preserved with the handoff.",
    status: "Tracked"
  }
]);

const PREVIEW_ROWS = Object.freeze([
  {
    title: "Verification report generated",
    meta: "Offline verifier output for independent inspection"
  },
  {
    title: "Hash invariants recomputed",
    meta: "Deterministic bindings for bytes and metadata"
  },
  {
    title: "Merkle root confirmed",
    meta: "Compact integrity anchor verified"
  },
  {
    title: "Rights gate applied",
    meta: "Usage posture and release control enforced"
  }
]);

const PREVIEW_MODULES = Object.freeze([
  { label: "Verifier", value: "Passed" },
  { label: "Hash chain", value: "Bound" },
  { label: "Merkle", value: "Computed" },
  { label: "Rights", value: "Review-gated" },
  { label: "Review", value: "Tracked" },
  { label: "Exports", value: "Ready" }
]);

const VERIFICATION_REPORT_EXCERPT = `{
  "verification_contract_version": "tp.meta.verification_report.v1",
  "verification_status": {
    "passed": true,
    "failure_code_label": null
  },
  "computed": {
    "metadata_entry_count": 14,
    "provenance_entry_count": 14,
    "provenance_merkle_root": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  }
}`;

function renderAnchor(href, label, className = "homepage-nav-link") {
  return `<a class="${className}" href="${escapeHtml(href)}">${escapeHtml(label)}</a>`;
}

function renderList(items, renderItem) {
  return items.map((item, index) => renderItem(item, index)).join("");
}

function renderHighlightCard(item) {
  return `<article class="highlight-card">
    <h3>${escapeHtml(item.title)}</h3>
    <p>${escapeHtml(item.detail)}</p>
  </article>`;
}

function renderHeroRouteCard(item) {
  return `<a class="hero-route-card" href="${escapeHtml(item.href)}" data-ui="${escapeHtml(item.dataUi)}">
    <p class="hero-route-kicker">${escapeHtml(item.kicker)}</p>
    <p class="hero-route-title">${escapeHtml(item.title)}</p>
    <p class="hero-route-detail">${escapeHtml(item.detail)}</p>
  </a>`;
}

function renderOutcomeCard(item) {
  return `<article class="feature-card">
    <h3>${escapeHtml(item.title)}</h3>
    <p>${escapeHtml(item.description)}</p>
  </article>`;
}

function renderEvidenceCard(item) {
  return `<article class="evidence-card">
    <p class="card-kicker">${escapeHtml(item.title)}</p>
    <p>${escapeHtml(item.description)}</p>
  </article>`;
}

function renderWorkflowCard(item) {
  return `<li>
    <article class="workflow-card">
      <div class="workflow-head">
        <span class="workflow-step">${escapeHtml(item.number)}</span>
        <h3>${escapeHtml(item.title)}</h3>
      </div>
      <p>${escapeHtml(item.description)}</p>
      <div class="workflow-output">
        <p class="workflow-output-label">Evidence output</p>
        <p class="workflow-output-value">${escapeHtml(item.output)}</p>
      </div>
    </article>
  </li>`;
}

function renderAudienceCard(item) {
  return `<article class="audience-card">
    <h3>${escapeHtml(item.title)}</h3>
    <p>${escapeHtml(item.summary)}</p>
    <ul>
      ${renderList(item.bullets, (bullet) => `<li>${escapeHtml(bullet)}</li>`)}
    </ul>
  </article>`;
}

function renderPrincipleCard(item) {
  return `<article class="principle-card">
    <h3>${escapeHtml(item.title)}</h3>
    <p>${escapeHtml(item.description)}</p>
  </article>`;
}

function renderFaqItem(item, index) {
  const openAttr = index === 0 ? " open" : "";
  return `<details class="faq-item"${openAttr}>
    <summary>${escapeHtml(item.question)}</summary>
    <p>${escapeHtml(item.answer)}</p>
  </details>`;
}

function renderArtifactRow(item) {
  return `<div class="artifact-row">
    <div class="artifact-copy">
      <p class="artifact-name">${escapeHtml(item.name)}</p>
      <p class="artifact-detail">${escapeHtml(item.detail)}</p>
    </div>
    <span class="artifact-status">${escapeHtml(item.status)}</span>
  </div>`;
}

function renderPreviewModule(item) {
  return `<div class="status-card">
    <p class="status-label">${escapeHtml(item.label)}</p>
    <p class="status-value">${escapeHtml(item.value)}</p>
  </div>`;
}

function renderPreviewRow(item) {
  return `<div class="checkpoint-row">
    <div class="checkpoint-marker" aria-hidden="true"></div>
    <div>
      <p class="checkpoint-title">${escapeHtml(item.title)}</p>
      <p class="checkpoint-meta">${escapeHtml(item.meta)}</p>
    </div>
  </div>`;
}

function renderReleaseBundlePreview() {
  return `<section class="bundle-preview" aria-labelledby="bundle-preview-title">
    <div class="bundle-preview-head">
      <div class="window-dots" aria-hidden="true">
        <span></span>
        <span></span>
        <span></span>
      </div>
      <span class="review-badge">Ready for review</span>
      <p class="section-kicker">Release bundle</p>
      <h2 id="bundle-preview-title">Asset R-2148</h2>
      <p class="bundle-intro">
        Prepared for legal, brand, and distribution review with a machine-readable verification report, rights posture, and operator evidence already attached.
      </p>
    </div>

    <div class="bundle-grid">
      <div class="bundle-column">
        <p class="subsection-kicker">Included artifacts</p>
        <div class="artifact-list">
          ${renderList(PREVIEW_ARTIFACTS, renderArtifactRow)}
        </div>
      </div>
      <div class="bundle-column bundle-column--status">
        <p class="subsection-kicker">Bundle status</p>
        <div class="status-grid">
          ${renderList(PREVIEW_MODULES, renderPreviewModule)}
        </div>
      </div>
    </div>

    <div class="bundle-review">
      <p class="subsection-kicker">Review checkpoints</p>
      <div class="checkpoint-list">
        ${renderList(PREVIEW_ROWS, renderPreviewRow)}
      </div>
    </div>

    <div id="proof-report" class="report-panel">
      <p class="subsection-kicker">Verification report excerpt</p>
      <pre><code>${escapeHtml(VERIFICATION_REPORT_EXCERPT)}</code></pre>
      <p class="report-note">Illustrative excerpt based on the verification contract. Example values shown.</p>
    </div>
  </section>`;
}

export function renderHomepage({ rumScript = "", scriptNonce = null } = {}) {
  const rumScriptTag = rumScript && scriptNonce
    ? `<script nonce="${escapeHtml(scriptNonce)}">${rumScript}</script>`
    : "";
  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Dynamic Neural Access</title>
    <link rel="stylesheet" href="/frontdoor-homepage.css" />
  </head>
  <body class="frontdoor-homepage" data-ui="homepage-shell">
    <a class="skip-link" href="#main-content">Skip to content</a>
    <div class="homepage-backdrop" aria-hidden="true">
      <video
        class="homepage-video"
        autoplay
        muted
        loop
        playsinline
        preload="metadata"
        disablePictureInPicture
        disableRemotePlayback
      >
        <source src="${FRONTDOOR_ASSETS.loopVideo}" type="video/mp4" />
      </video>
      <div class="homepage-backdrop__overlay"></div>
      <div class="homepage-backdrop__grid"></div>
      <div class="homepage-backdrop__glow homepage-backdrop__glow--top"></div>
      <div class="homepage-backdrop__glow homepage-backdrop__glow--bottom"></div>
    </div>

    <header class="site-header" data-ui="homepage-header">
      <div class="site-header__inner">
        <a class="brand-lockup" href="/" aria-label="Dynamic Neural Access home">
          <span class="brand-lockup__mark">
            ${renderBrandAsset({
              kind: "symbol",
              variant: "dark",
              alt: "Dynamic Neural Access",
              className: "brand-asset brand-asset--symbol"
            })}
          </span>
          <span class="brand-lockup__copy">
            <span class="brand-lockup__kicker">Dynamic Neural Access</span>
            <span class="brand-lockup__title">Verifier-backed release proof for premium media.</span>
          </span>
        </a>

        <nav class="site-nav site-nav--desktop" aria-label="Primary navigation" data-ui="homepage-nav">
          ${renderList(NAV_LINKS, (item) => renderAnchor(`#${item.id}`, item.label))}
        </nav>

        <div class="site-actions">
          <a class="site-link" href="/login" data-ui="homepage-operator-link">Operator Login</a>
          <a class="site-cta site-cta--ghost" href="/login" data-ui="homepage-utility-cta">Operator Access</a>
        </div>

        <details class="site-mobile-menu">
          <summary>Menu</summary>
          <div class="site-mobile-menu__panel">
            <nav aria-label="Mobile navigation">
              ${renderList(NAV_LINKS, (item) => renderAnchor(`#${item.id}`, item.label, "homepage-mobile-link"))}
            </nav>
            <div class="site-mobile-menu__actions">
              <a class="site-link site-link--mobile" href="/login" data-ui="homepage-mobile-operator-link">Operator Login</a>
              <a class="site-cta site-cta--ghost site-cta--mobile" href="/login" data-ui="homepage-mobile-utility-cta">Operator Access</a>
            </div>
          </div>
        </details>
      </div>
    </header>

    <main id="main-content" class="homepage-main" data-ui="homepage-main">
      <section class="hero-section" aria-labelledby="hero-title" data-ui="homepage-hero">
        <div class="hero-copy">
          <div class="hero-lockup" data-ui="homepage-hero-lockup">
            ${renderBrandAsset({
              kind: "lockup",
              variant: "dark",
              alt: "Dynamic Neural Access",
              className: "hero-lockup__asset"
            })}
          </div>
          <p class="section-kicker">Dynamic Neural Access</p>
          <h1 id="hero-title" data-ui="homepage-hero-title">Make premium media verifiable before it ships.</h1>
          <p class="hero-lede" data-ui="homepage-hero-lede">
            Verifier-backed release proof for premium media. Keep provenance, rights posture, and operator history attached before anything leaves the workflow.
          </p>
          <div class="hero-actions" data-ui="homepage-hero-actions">
            <a class="site-cta" href="/login" data-ui="homepage-primary-cta">Operator Access</a>
            <a class="site-cta site-cta--secondary" href="#proof-report" data-ui="homepage-secondary-cta">Inspect Verification Report</a>
            <a class="hero-inline-link" href="#workflow" data-ui="homepage-learn-link">Explore workflow</a>
          </div>
          <p class="hero-access-note" data-ui="homepage-hero-note">
            Public proof stays visible. Managed entry opens only when operator work needs to continue.
          </p>
          <div class="entry-rail" data-ui="homepage-entry-rail">
            <article class="entry-card entry-card--proofband" data-state="public-proof">
              <div class="entry-card__summary">
                <p class="entry-card__kicker">Proof snapshot</p>
                <p class="entry-card__title">One release bundle, one decision path.</p>
                <p class="entry-card__detail">Verification report, provenance artifacts, rights posture, and operator review stay in one reviewable handoff.</p>
              </div>
              <div class="entry-card__meta">
                <div class="entry-card__meta-item">
                  <p class="entry-card__meta-label">Public proof</p>
                  <p class="entry-card__meta-value">Inspect the report before access is requested.</p>
                </div>
                <div class="entry-card__meta-item">
                  <p class="entry-card__meta-label">Managed entry</p>
                  <p class="entry-card__meta-value">Access verification stays separate from operator credentials.</p>
                </div>
                <div class="entry-card__meta-item">
                  <p class="entry-card__meta-label">Operator console</p>
                  <p class="entry-card__meta-value">Build, operate, and review continue inside the governed shell.</p>
                </div>
              </div>
            </article>
          </div>
          <div class="signal-strip" aria-label="Proof signals">
            ${renderList(HERO_SIGNALS, (item) => `<span class="signal-pill">${escapeHtml(item)}</span>`)}
          </div>
          <div class="guardrail-callout">
            <strong>Not a truth engine.</strong> A release-readiness system for provenance context, rights clarity, and reviewable evidence.
          </div>
        </div>
        ${renderReleaseBundlePreview()}
      </section>

      <section id="platform" class="homepage-section" aria-labelledby="platform-title">
        <div class="section-head">
          <p class="section-kicker">Why teams adopt DNA</p>
          <h2 id="platform-title">Certification with operational leverage.</h2>
          <p>
            The value is not abstract trust rhetoric. It is faster approval, cleaner licensing posture, and release bundles partners can inspect without chasing context across email, chat, and ad hoc notes.
          </p>
        </div>
        <div class="feature-grid">
          ${renderList(OUTCOME_CARDS, renderOutcomeCard)}
        </div>
      </section>

      <section id="proof" class="homepage-section homepage-section--proof" aria-labelledby="proof-title">
        <div class="proof-layout">
          <div class="proof-copy">
            <div class="section-head">
              <p class="section-kicker">What ships</p>
              <h2 id="proof-title">Every certified asset leaves with a verification report.</h2>
              <p>
                Instead of handing off screenshots and memory, DNA assembles a machine-readable verification report, supporting provenance artifacts, rights posture, and release history into one inspectable handoff.
              </p>
            </div>
            <aside class="standards-note">
              <p class="subsection-kicker">Standards note</p>
              <p>
                When enabled, DNA can emit Content Credentials-compatible manifests. Optional timestamp and signature attachments can be added when configured. If downstream platforms strip metadata, the workflow record still preserves reviewable evidence and controlled release history.
              </p>
            </aside>
          </div>
          <div class="evidence-grid">
            ${renderList(EVIDENCE_ITEMS, renderEvidenceCard)}
          </div>
        </div>
      </section>

      <section id="workflow" class="homepage-section" aria-labelledby="workflow-title">
        <div class="section-head">
          <p class="section-kicker">Workflow</p>
          <h2 id="workflow-title">Verify. Enhance. Enforce. Distribute.</h2>
          <p>
            A disciplined release path for premium media, with each stage producing artifacts that make the next decision easier to audit.
          </p>
        </div>
        <ol class="workflow-grid" role="list">
          ${renderList(WORKFLOW_STEPS, renderWorkflowCard)}
        </ol>
      </section>

      <section id="teams" class="homepage-section" aria-labelledby="teams-title">
        <div class="section-head">
          <p class="section-kicker">Teams</p>
          <h2 id="teams-title">Built for organizations that cannot afford ambiguity.</h2>
          <p>
            From single creators to multi-party distribution environments, DNA is framed around the places where provenance, rights, and release risk collide.
          </p>
        </div>
        <div class="audience-grid">
          ${renderList(AUDIENCE_SEGMENTS, renderAudienceCard)}
        </div>
      </section>

      <section class="homepage-section homepage-section--panel" aria-labelledby="principles-title">
        <div class="section-head">
          <p class="section-kicker">Standards posture</p>
          <h2 id="principles-title">Precise enough for counsel. Clear enough for operators.</h2>
          <p>
            DNA is strongest when it explains what is certified, what can be exported, and where downstream platform limits still apply.
          </p>
        </div>
        <div class="principles-grid">
          ${renderList(PRINCIPLES, renderPrincipleCard)}
        </div>
      </section>

      <section id="faq" class="homepage-section" aria-labelledby="faq-title">
        <div class="section-head">
          <p class="section-kicker">FAQ</p>
          <h2 id="faq-title">The sharp questions, answered early.</h2>
          <p>
            These are the questions serious buyers ask before they trust a release-control layer with high-value media.
          </p>
        </div>
        <div class="faq-grid">
          ${renderList(FAQ_ITEMS, renderFaqItem)}
        </div>
      </section>

      <section class="homepage-section homepage-section--cta" aria-labelledby="cta-title" data-ui="homepage-final-cta">
        <div class="cta-panel">
          <div class="section-head">
            <p class="section-kicker">Next move</p>
            <h2 id="cta-title">Bring certification to every asset that leaves your pipeline.</h2>
            <p>
              Choose the public proof path when you need orientation, or continue into managed operator access when you are ready to run governed work.
            </p>
          </div>
          <div class="hero-actions" data-ui="homepage-final-actions">
            <a class="site-cta" href="/login" data-ui="homepage-final-primary-cta">Open Operator Access</a>
            <a class="site-cta site-cta--secondary" href="#workflow" data-ui="homepage-final-secondary-cta">Explore Workflow</a>
          </div>
        </div>
      </section>
    </main>

    <footer class="site-footer">
      <div class="site-footer__inner">
        <div>
          <p class="site-footer__title">Dynamic Neural Access</p>
          <p class="site-footer__copy">Verifier-backed release proof for premium media.</p>
        </div>
        <a class="site-link" href="/login" data-ui="homepage-footer-login">Operator Login</a>
      </div>
    </footer>
    ${rumScriptTag}
  </body>
</html>`;
}
