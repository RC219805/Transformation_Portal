import { NextResponse } from "next/server.js";

import { getSessionFromRequest } from "../lib/sessions.js";
import { applySecurityHeaders, FRONTDOOR_CSP } from "../lib/http.js";
import { escapeHtml, FRONTDOOR_ASSETS, renderBrandAsset } from "../lib/brand.js";

export const runtime = "nodejs";

function renderHomepage({ operatorHref, operatorLabel }) {
  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Dynamic Neural Access</title>
    <link rel="stylesheet" href="/login.css" />
  </head>
  <body class="frontdoor-homepage">
    <main class="shell">
      <video
        class="hero-video hero-video--homepage"
        autoplay
        muted
        loop
        playsinline
        preload="metadata"
        disablePictureInPicture
        disableRemotePlayback
        poster=""
        aria-hidden="true"
      >
        <source src="${FRONTDOOR_ASSETS.loopVideo}" type="video/mp4" />
      </video>
      <div class="homepage-noise" aria-hidden="true"></div>
      <section class="homepage-content">
        <header class="site-header">
          <a class="brand-lockup" href="/" aria-label="Dynamic Neural Access home">
            <span class="brand-asset-frame brand-asset-frame--header">
              ${renderBrandAsset({
                variant: "dark",
                alt: "Dynamic Neural Access",
                className: "brand-asset brand-asset--header"
              })}
            </span>
            <span class="brand-copy">
              <span class="brand-kicker">Dynamic Neural Access</span>
              <span class="brand-title">Certified premium media, governed from first proof to final distribution.</span>
            </span>
          </a>
          <nav class="site-actions" aria-label="Primary">
            <a class="site-link" href="/login">Operator Login</a>
            <a class="site-cta site-cta--ghost" href="${escapeHtml(operatorHref)}">${escapeHtml(operatorLabel)}</a>
          </nav>
        </header>

        <section class="homepage-hero">
          <div class="hero-copy">
            <p class="eyebrow">Dynamic Neural Access</p>
            <h1>Certified Premium Media for the AI Era</h1>
            <p class="lede">
              Dynamic Neural Access certifies and enhances media into trusted, premium-ready assets by verifying authenticity, ownership, and provenance, then optimizing quality, format, and compliance for secure distribution and monetization.
            </p>
            <p class="one-line-pitch">Dynamic Neural Access turns media into certified, premium-ready assets for secure distribution and monetization.</p>
            <div class="hero-actions">
              <a class="site-cta" href="/login">Certify Your Media</a>
              <a class="site-cta site-cta--secondary" href="#how-it-works">See How It Works</a>
            </div>
          </div>
          <aside class="hero-proof">
            <p class="hero-proof-label">Investor description</p>
            <p class="hero-proof-text">
              Dynamic Neural Access certifies and enhances media for the AI era, turning content into trusted, premium-ready assets with governance, licensing, and traceability built in from the start.
            </p>
            <div class="proof-stack">
              <div>
                <span class="proof-chip">Authenticity</span>
                <span class="proof-chip">Ownership</span>
                <span class="proof-chip">Provenance</span>
              </div>
              <div>
                <span class="proof-chip">Licensing</span>
                <span class="proof-chip">Compliance</span>
                <span class="proof-chip">Traceability</span>
              </div>
            </div>
          </aside>
        </section>

        <section class="trust-strip" aria-label="Trust foundations">
          <span>Authenticity</span>
          <span>Provenance</span>
          <span>Ownership</span>
          <span>Licensing</span>
          <span>Compliance</span>
        </section>

        <section id="how-it-works" class="homepage-band">
          <div class="section-head">
            <p class="eyebrow">How it works</p>
            <h2>Verify. Enhance. Govern. Distribute.</h2>
          </div>
          <div class="editorial-grid">
            <article>
              <h3>Verify</h3>
              <p>Establish authenticity, ownership, and provenance before an asset enters circulation.</p>
            </article>
            <article>
              <h3>Enhance</h3>
              <p>Optimize quality, format, and readiness for premium publishing, licensing, and AI-era reuse.</p>
            </article>
            <article>
              <h3>Govern</h3>
              <p>Attach licensing controls, operational policy, and auditable run contracts to every certified output.</p>
            </article>
            <article>
              <h3>Distribute</h3>
              <p>Release premium-ready media with traceability and confidence across secure downstream channels.</p>
            </article>
          </div>
        </section>

        <section class="homepage-band homepage-band--split">
          <div class="section-head">
            <p class="eyebrow">Who it is for</p>
            <h2>Built for high-value media ecosystems.</h2>
          </div>
          <div class="use-case-list">
            <article>
              <h3>Creators</h3>
              <p>Protect authorship, certify outputs, and deliver premium-ready assets for licensing and monetization.</p>
            </article>
            <article>
              <h3>Studios</h3>
              <p>Govern production pipelines with consistent provenance, traceability, and reviewable outputs.</p>
            </article>
            <article>
              <h3>Festivals</h3>
              <p>Preserve authenticity and chain of custody across intake, curation, exhibition, and archival handoff.</p>
            </article>
            <article>
              <h3>Brands</h3>
              <p>Deploy governed media with clear rights posture, compliance context, and premium finish standards.</p>
            </article>
          </div>
        </section>

        <section class="homepage-band homepage-proof-band">
          <div class="section-head">
            <p class="eyebrow">Governance</p>
            <h2>Certification that carries operational meaning.</h2>
          </div>
          <div class="proof-rail">
            <article class="proof-panel">
              <p class="proof-panel-kicker">Licensing control</p>
              <h3>Operational guardrails stay attached to the asset.</h3>
              <p>Rights acknowledgments, governed presets, and operator posture remain visible at dispatch and review time.</p>
            </article>
            <article class="proof-panel">
              <p class="proof-panel-kicker">Traceability</p>
              <h3>Every reviewed output keeps a readable chain of custody.</h3>
              <p>Run cards, manifests, and indexed outputs make provenance and distribution readiness legible in seconds.</p>
            </article>
            <article class="proof-panel">
              <p class="proof-panel-kicker">Product proof</p>
              <h3>Before and after treatment becomes an auditable premium surface.</h3>
              <p>Compare the certified output, inspect the provenance card, and hand off governed artifacts without leaving the review flow.</p>
            </article>
          </div>
        </section>

        <section id="final-cta" class="homepage-band homepage-band--cta">
          <div class="section-head">
            <p class="eyebrow">Next move</p>
            <h2>Bring governed certification to the media you distribute.</h2>
          </div>
          <p class="cta-copy">Start operator access for secure orchestration, or open the console if you already hold a governed session.</p>
          <div class="hero-actions">
            <a class="site-cta" href="/login">Operator Login</a>
            <a class="site-cta site-cta--secondary" href="${escapeHtml(operatorHref)}">${escapeHtml(operatorLabel)}</a>
          </div>
        </section>
      </section>
    </main>
  </body>
</html>`;
}

export async function GET(request) {
  const session = getSessionFromRequest(request, { touch: false });
  const html = renderHomepage({
    operatorHref: session?.authenticated ? "/portal" : "/login",
    operatorLabel: session?.authenticated ? "Open Console" : "Secure Access"
  });
  const response = new NextResponse(html, {
    status: 200,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store"
    }
  });
  return applySecurityHeaders(response, { csp: FRONTDOOR_CSP });
}
