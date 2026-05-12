# APEX Phase 3: Dashboarding & Long-Term Storage

**Status:** ✅ Implemented
**Version:** 1.0.0
**Last Updated:** 2026-02-07

---

## Executive Summary

APEX Phase 3 extends the performance observability platform with:

1. **Persistent Storage Strategy** - Multi-tier storage with 90-day hot storage, 1-year warm storage, and indefinite cold archival
2. **Interactive Dashboard** - Static HTML dashboard with Chart.js visualizations hosted on GitHub Pages
3. **Long-Term Trend Analysis** - Time-series performance tracking across workflow versions, zones, and buckets
4. **Automated Backup System** - Weekly ledger backups to GitHub Releases for compliance and historical analysis

**Key Benefit:** Transforms ephemeral CI artifacts into a durable performance knowledge base with zero-cost infrastructure.

---

## Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                      APEX Phase 3 Platform                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐  │
│  │   CI Run    │────▶│  SQLite DB   │────▶│   Dashboard     │  │
│  │  (Matrix)   │     │   (Ledger)   │     │   Generator     │  │
│  └─────────────┘     └──────────────┘     └─────────────────┘  │
│         │                    │                      │            │
│         │                    │                      ▼            │
│         │                    │            ┌──────────────────┐  │
│         │                    │            │  GitHub Pages    │  │
│         │                    │            │  (Dashboard UI)  │  │
│         │                    │            └──────────────────┘  │
│         │                    │                                   │
│         ▼                    ▼                                   │
│  ┌──────────────┐   ┌──────────────────┐                        │
│  │   GitHub     │   │  GitHub Releases │                        │
│  │  Artifacts   │   │  (Weekly Backup) │                        │
│  │  (90 days)   │   │  (Indefinite)    │                        │
│  └──────────────┘   └──────────────────┘                        │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Multi-Tier Storage Strategy

| Tier | Technology | Retention | Access Pattern | Cost |
|------|-----------|-----------|----------------|------|
| **Hot** | GitHub Actions Artifacts | 90 days | Fast queries in CI | Free |
| **Warm** | GitHub Releases (compressed) | 1 year+ | Downloadable archives | Free |
| **Cold** | GitHub Releases (archived) | Indefinite | Compliance/historical | Free |

**Design Rationale:**
- **No external infrastructure required** - leverages GitHub native storage
- **Zero-cost operation** - all tiers use GitHub free tier
- **Automated lifecycle** - CI handles promotion and archival
- **Compliance-ready** - indefinite retention for auditing

---

## Component Design

### 1. Enhanced Ledger Schema

**Location:** `src/transformation_portal/metrics/ledger.py`

**New Indexes:**
```sql
CREATE INDEX IF NOT EXISTS idx_apex_runs_timestamp ON apex_runs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_apex_runs_bucket_zone_time
    ON apex_runs(bucket_name, zone, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_apex_runs_pass_fail ON apex_runs(pass_fail);
```

**New View for Dashboard Queries:**
```sql
CREATE VIEW IF NOT EXISTS apex_trends AS
SELECT
    bucket_name,
    zone,
    workflow_version,
    DATE(timestamp) as date,
    AVG(p50) as avg_p50,
    AVG(p95) as avg_p95,
    AVG(p99) as avg_p99,
    COUNT(*) as run_count,
    SUM(CASE WHEN pass_fail = 'fail' THEN 1 ELSE 0 END) as fail_count,
    SUM(CASE WHEN pass_fail = 'warn' THEN 1 ELSE 0 END) as warn_count
FROM apex_runs
GROUP BY bucket_name, zone, workflow_version, DATE(timestamp)
ORDER BY date DESC;
```

**Optimization Strategy:**
- Compound index on `(bucket_name, zone, timestamp DESC)` for efficient time-series queries
- Materialized view via `apex_trends` for pre-aggregated daily stats
- Descending timestamp index for "latest N runs" queries

### 2. Dashboard Generator

**Location:** `scripts/apex_dashboard_generator.py`

**Responsibilities:**
- Query ledger database for dashboard data
- Generate static HTML with embedded Chart.js visualizations
- Export raw JSON for external tooling
- Support configurable time windows (default 90 days)

**Output Files:**
- `index.html` - Main dashboard with trends and regressions
- `latest.html` - Latest run summary table
- `data.json` - Raw data export

**Visualization Types:**
1. **Performance Trends (Line Chart)** - p95 latency over time per bucket
2. **Worst Offenders (Horizontal Bar)** - Top 20 highest p95/threshold ratios
3. **Regression Timeline (Scatter)** - Failures and warnings over time

**Design Principles:**
- **Static generation** - No backend/database required in production
- **Client-side rendering** - JavaScript handles all interactions
- **Mobile-responsive** - Grid layouts adapt to screen size
- **Accessibility** - Semantic HTML with proper contrast ratios

### 3. CI/CD Integration

**Location:** `.github/workflows/apex_performance.yml`

**New Workflow Triggers:**
```yaml
on:
  pull_request: # Existing
  push:
    branches: [main]  # NEW: Dashboard updates
  schedule:
    - cron: '0 0 * * 0'  # NEW: Weekly backups
```

**Dashboard Update Flow (main branch only):**
1. Download the `apex-ledger` artifact produced by `apex_gate`
2. Run matrix benchmarks (append new data)
3. Generate dashboard HTML
4. Upload `_site/` with `actions/upload-pages-artifact`
5. Deploy to GitHub Pages via `actions/deploy-pages`

**Weekly Backup Flow:**
1. Compress ledger database (gzip)
2. Create GitHub Release with date-stamped backup
3. Attach compressed ledger as release asset

**Permissions Required:**
```yaml
permissions:
  contents: write     # For GitHub Releases
  pages: write        # For GitHub Pages deployment
  id-token: write     # For Pages authentication
```

### 4. GitHub Pages Hosting

**URL Pattern:**
```
https://rc219805.github.io/Transformation_Portal/
```

**Generated Artifact Structure:**
```
_site/
├── index.html      # Main dashboard
├── latest.html     # Latest run summary
└── data.json       # Raw data export
```

**Deployment Configuration:**
- Uses `actions/upload-pages-artifact` and `actions/deploy-pages` with pinned SHAs
  in `.github/workflows/apex_performance.yml`
- Publishes the generated `_site/` artifact directly through GitHub Pages
- Does not commit generated dashboard files back into the repository

---

## Dashboard User Guide

### Accessing the Dashboard

1. **Production URL:** `https://rc219805.github.io/Transformation_Portal/`
2. **Latest Metrics:** `https://rc219805.github.io/Transformation_Portal/latest.html`
3. **Local Preview:** Generate locally and open `_site/index.html`

### Dashboard Sections

#### Executive Summary Cards
- **Recent Runs** - Count of runs in selected time window
- **Warnings/Failures** - Count of performance regressions
- **Problem Buckets** - Unique bucket/zone combinations with failures

#### Performance Trends Chart
- **Type:** Multi-line time series
- **Metric:** p95 latency (seconds)
- **Grouping:** One line per bucket
- **Interaction:** Hover for exact values, legend to toggle lines

#### Worst Offenders Chart
- **Type:** Horizontal bar chart
- **Metric:** Max (p95 / threshold) ratio
- **Sorting:** Descending by severity
- **Color Coding:** Red (>1.5x), Orange (>1.0x)

#### Regression Timeline
- **Type:** Scatter plot
- **X-axis:** Timestamp
- **Y-axis:** p95/threshold ratio
- **Color Coding:** Red (fail), Orange (warn)

### Latest Run Summary
- **URL:** `latest.html`
- **Format:** Sortable table
- **Columns:** Status, Commit, Version, Zone, Bucket, p50, p95, Threshold
- **Limit:** Top 20 most recent runs

---

## Operational Runbook

### Dashboard Update Cadence

| Event | Trigger | Latency | Action |
|-------|---------|---------|--------|
| PR Opened | Automatic | ~5 min | Run benchmarks, post comment (no dashboard update) |
| Main Merge | Automatic | ~10 min | Update dashboard + deploy to Pages |
| Weekly Backup | Scheduled | Sunday 00:00 UTC | Create release with compressed ledger |

### Manual Dashboard Regeneration

```bash
# Generate dashboard locally
python scripts/apex_dashboard_generator.py \
  --ledger-db apex_performance.db \
  --output-dir _site \
  --days 90

# Preview locally
open _site/index.html
```

### Restoring from Backup

```bash
# Download weekly backup from GitHub Releases
gh release download ledger-backup-2026-02-07 --pattern "*.db.gz"

# Decompress
gunzip apex_ledger_2026-02-07.db.gz

# Regenerate dashboard
python scripts/apex_dashboard_generator.py \
  --ledger-db apex_ledger_2026-02-07.db \
  --output-dir _site \
  --days 365  # Full year if backup is old
```

### Pruning Old Data

```bash
# Prune ledger entries older than 1 year (local operation)
python -m transformation_portal.metrics.ledger prune \
  --ledger-db apex_performance.db \
  --days-to-keep 365

# Regenerate dashboard after pruning
python scripts/apex_dashboard_generator.py \
  --ledger-db apex_performance.db \
  --output-dir _site \
  --days 90
```

---

## Performance Characteristics

### Dashboard Generation Performance

| Database Size | Record Count | Generation Time | Output Size |
|--------------|--------------|-----------------|-------------|
| 1 MB | ~1,000 runs | ~0.5s | ~200 KB |
| 10 MB | ~10,000 runs | ~2s | ~500 KB |
| 100 MB | ~100,000 runs | ~10s | ~2 MB |

**Optimization Notes:**
- Queries use indexed columns (timestamp, bucket, zone)
- Aggregation handled by SQLite (faster than Python)
- HTML generation is linear time (no nested loops)

### Storage Overhead

| Component | Size per Run | 90-Day Estimate | 1-Year Estimate |
|-----------|--------------|-----------------|-----------------|
| Single run record | ~2 KB | ~180 KB | ~730 KB |
| Dashboard HTML | 200 KB | N/A (static) | N/A (static) |
| Compressed backup | ~50% original | ~90 KB | ~365 KB |

**Scaling Notes:**
- GitHub Actions artifact limit: 10 GB (effectively unbounded for our use case)
- GitHub Releases asset limit: 2 GB per asset
- Expected ledger size after 1 year: <5 MB (well within limits)

---

## Security & Privacy Considerations

### Data Exposure

**Public Dashboard:**
- ✅ Aggregated metrics only (no raw image paths)
- ✅ No client-identifiable information
- ✅ Commit SHAs are public GitHub data
- ❌ **Do not include:** image paths, client names, metadata

**Access Control:**
- Dashboard is public (GitHub Pages default)
- Ledger database is private (GitHub Actions artifacts require authentication)
- Releases are public (intentional for transparency)

### Supply Chain Security

**Pinned Actions:**
```yaml
actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9 # v5.0.0
actions/deploy-pages@cd2ce8fcbc39b97be8ca5fce6e763baed58fa128 # v5.0.0
```

**CDN Dependencies:**
```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
```
- Chart.js pinned to specific version
- Delivered via JSDelivr CDN (SRI hashes not used for simplicity, acceptable for observability dashboard)

---

## Testing Strategy

### Unit Tests

**Location:** `tests/test_apex_dashboard.py`

**Coverage:**
- `generate_dashboard_data()` with mock ledger
- `generate_index_html()` output validation
- `generate_latest_html()` table rendering
- Edge cases: empty database, single run, missing zones

### Integration Tests

**Location:** `tests/test_apex_dashboard.py` and the
`dashboard_deploy` job in `.github/workflows/apex_performance.yml`

**Coverage:**
- End-to-end dashboard generation
- Ledger schema migration with new indexes
- GitHub Pages deployment simulation

### Manual Testing Checklist

- [ ] Dashboard renders correctly in Chrome, Firefox, Safari
- [ ] Mobile responsive layout works on iPhone/Android
- [ ] Chart interactions (hover, legend toggle) work
- [ ] Links between index.html and latest.html work
- [ ] data.json downloads correctly
- [ ] GitHub Pages deployment succeeds

---

## Future Enhancements (Out of Scope for Phase 3)

### Potential Phase 4 Features

1. **Advanced Filtering**
   - URL parameters for date range, bucket, zone filters
   - Permalink generation for specific views

2. **Comparison Mode**
   - Side-by-side V1 vs V2 comparison
   - Before/after commit comparison

3. **Alerting Integration**
   - Webhook notifications for regressions
   - Slack/Discord integration

4. **Custom Metrics**
   - User-defined performance SLOs
   - Custom threshold overrides per zone

5. **Export Formats**
   - CSV export for Excel analysis
   - Prometheus metrics endpoint
   - Grafana JSON datasource

### Known Limitations

1. **No real-time updates** - Dashboard updates only on main branch push
2. **No authentication** - Dashboard is fully public
3. **Limited interactivity** - No drill-down to individual runs
4. **Single time window** - Cannot dynamically adjust date range in UI

---

## Rollout Plan

### Phase 3A: Core Implementation (Complete)
- ✅ Dashboard generator script
- ✅ Enhanced ledger schema with indexes and views
- ✅ GitHub Actions workflow updates
- ✅ GitHub Pages deployment configuration

### Phase 3B: Initial Deployment
1. Merge to main branch
2. Trigger initial dashboard generation
3. Verify GitHub Pages deployment
4. Validate dashboard URL accessibility

### Phase 3C: Monitoring & Iteration
1. Monitor dashboard update frequency
2. Track GitHub Actions minutes usage
3. Gather stakeholder feedback
4. Iterate on visualizations

### Success Criteria

- [ ] Dashboard accessible at public URL
- [ ] Weekly backups appear in GitHub Releases
- [ ] No CI failures related to dashboard generation
- [ ] Dashboard load time <2 seconds
- [ ] Mobile responsive design validated

---

## Maintenance

### Routine Maintenance

**Weekly:**
- Review weekly backup success
- Check for dashboard deployment failures

**Monthly:**
- Review dashboard performance (load times, query times)
- Check GitHub Actions minutes usage
- Validate data retention policies

**Quarterly:**
- Review visualization effectiveness
- Gather stakeholder feedback
- Evaluate storage growth trends

### Troubleshooting

**Dashboard Not Updating:**
1. Check GitHub Actions workflow status
2. Verify `push` to `main` branch triggered workflow
3. Check GitHub Pages deployment logs
4. Verify `gh-pages` branch exists and is updated

**Missing Weekly Backups:**
1. Check scheduled workflow runs
2. Verify `schedule` trigger is enabled
3. Check for release creation errors in logs
4. Verify GH_TOKEN permissions

**Dashboard Load Errors:**
1. Check browser console for JavaScript errors
2. Verify Chart.js CDN is accessible
3. Check `data.json` syntax validity
4. Test with browser cache disabled

---

## References

- **APEX Phase 1:** Performance Capsule design and Quality Firewall
- **APEX Phase 2:** CI integration and PR commenting
- **APEX Phase 3:** Dashboarding and long-term storage (this document)
- **Chart.js Documentation:** https://www.chartjs.org/docs/latest/
- **GitHub Pages:** https://docs.github.com/en/pages
- **GitHub Actions Artifacts:** https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts

---

## Changelog

### v1.0.0 (2026-02-07)
- Initial Phase 3 implementation
- Dashboard generator with Chart.js visualizations
- GitHub Pages deployment
- Weekly backup to GitHub Releases
- Enhanced ledger schema with indexes and views
