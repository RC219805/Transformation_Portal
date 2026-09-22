// Overview-only capability catalog. The initial Build/Operate/Review bundle
// does not need this catalog; resolve its context when rendering after load.
import { buildPortalCapabilityCatalog, normalizeCapabilityStatus } from "./internal/capabilities.js";

export function createDeferredOverviewSurfaceApi(host) {
    const { els, getCapabilityContext, titleCaseToken } = host;
    function renderCapabilityMatrix(payload) {
        if (!els.capabilityMatrix) return;
        const catalog = buildPortalCapabilityCatalog(getCapabilityContext(payload));
        if (els.capabilitySummaryBadge) {
            const enabledCount = Number(catalog.summary?.enabled) || 0;
            const availableCount = Number(catalog.summary?.available) || 0;
            const nextStatus = normalizeCapabilityStatus(catalog.summary?.nextActionStatus, 'available');
            els.capabilitySummaryBadge.textContent = `${enabledCount} active · ${availableCount} available`;
            els.capabilitySummaryBadge.dataset.capabilityStatus = nextStatus;
        }
        if (els.capabilitySummaryDetail) {
            els.capabilitySummaryDetail.textContent = catalog.summary?.nextActionLabel || 'Capability catalog is ready.';
        }

        const fragment = document.createDocumentFragment();
        const otherFragment = document.createDocumentFragment();
        catalog.rows.forEach((capability) => {
            const row = document.createElement('article');
            row.className = 'capability-row';
            row.dataset.ui = 'capability-row';
            row.dataset.capabilityId = String(capability.id || '');
            row.dataset.capabilityStatus = normalizeCapabilityStatus(capability.status, 'available');
            row.setAttribute('role', 'listitem');

            const header = document.createElement('div');
            header.className = 'capability-row__header';

            const copy = document.createElement('div');
            copy.className = 'capability-row__copy';

            const group = document.createElement('p');
            group.className = 'capability-row__group';
            group.textContent = capability.group || 'Portal';

            const label = document.createElement('p');
            label.className = 'capability-row__label';
            label.textContent = capability.label || capability.id || 'Capability';

            copy.append(group, label);

            const badge = document.createElement('span');
            badge.className = 'capability-row__status';
            badge.textContent = capability.statusLabel || titleCaseToken(capability.status, 'Available');

            header.append(copy, badge);

            const summary = document.createElement('p');
            summary.className = 'capability-row__summary';
            summary.textContent = capability.summary || '';

            const detail = document.createElement('p');
            detail.className = 'capability-row__detail';
            detail.textContent = capability.detail || '';

            row.append(header, summary, detail);
            if (capability.scope === 'current') fragment.appendChild(row);
            else otherFragment.appendChild(row);
        });

        els.capabilityMatrix.replaceChildren(fragment);
        if (els.capabilityOtherMatrix) els.capabilityOtherMatrix.replaceChildren(otherFragment);
        if (els.capabilityOtherSummary) {
            els.capabilityOtherSummary.textContent = `Other workflows and tools (${catalog.summary.outsideWorkflow})`;
        }
    }

    return { renderCapabilityMatrix };
}
