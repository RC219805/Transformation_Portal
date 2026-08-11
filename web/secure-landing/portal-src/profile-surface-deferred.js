// Deferred Build profile manager.
//
// Saved profiles are local browser state and only matter on the Build surface.
// Keeping the dialog, validation, and destructive-action confirmations here
// avoids growing the bootstrap bundle for operators who never open Build.

const PROFILE_NAME_MAX_LENGTH = 64;
const RESERVED_PROFILE_NAMES = new Set(['__proto__', 'constructor', 'prototype']);

export function createDeferredProfileSurfaceApi(host) {
    const {
        state,
        els,
        storageKey,
        ownerKey,
        legacyStorageKey,
        unprofiledBaselineRecord,
        restoredUnprofiledDraft,
        announcePortalStatus,
        copyDraftConfig,
        createToast,
        fetchPresetsForPipeline,
        persistTransientDraft,
        rememberOverlayTrigger,
        restoreOverlayFocus,
        setPortalBackgroundInert,
        updateUIFromState,
    } = host;

    let refs = null;
    let pendingLoadName = '';
    let protectRestoredUnprofiledDraft = Boolean(restoredUnprofiledDraft);

    function normalizeName(value) {
        return String(value || '').trim();
    }

    function nameValidationMessage(value) {
        const name = normalizeName(value);
        if (!name) return 'Enter a profile name.';
        if (name.length > PROFILE_NAME_MAX_LENGTH) return `Profile names must be ${PROFILE_NAME_MAX_LENGTH} characters or fewer.`;
        if (/[\u0000-\u001f\u007f]/.test(name)) return 'Profile names cannot contain control characters.';
        if (RESERVED_PROFILE_NAMES.has(name.toLowerCase())) return 'Choose a profile name that is not a reserved object key.';
        return '';
    }

    function hasProfile(profiles, name) {
        return Boolean(name && profiles && Object.prototype.hasOwnProperty.call(profiles, name));
    }

    function resolvedOwnerKey() {
        return normalizeName(typeof ownerKey === 'function' ? ownerKey() : ownerKey);
    }

    function scopedStorageKey() {
        const owner = resolvedOwnerKey();
        return owner ? `${storageKey}:${encodeURIComponent(owner)}` : '';
    }

    function legacyStorageKeys() {
        const extraKey = normalizeName(typeof legacyStorageKey === 'function' ? legacyStorageKey() : legacyStorageKey);
        return extraKey && extraKey !== storageKey ? [storageKey, extraKey] : [storageKey];
    }

    function parseProfiles(rawValue) {
        const parsed = JSON.parse(rawValue || '{}');
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return Object.create(null);
        return Object.assign(Object.create(null), Object.fromEntries(Object.entries(parsed).filter(([name, profile]) => (
            !nameValidationMessage(name)
            && profile
            && typeof profile === 'object'
            && !Array.isArray(profile)
            && typeof profile.pipeline === 'string'
            && profile.config
            && typeof profile.config === 'object'
            && !Array.isArray(profile.config)
        ))));
    }

    function getLegacyProfiles() {
        return legacyStorageKeys().reduce((profiles, key) => {
            try {
                Object.assign(profiles, parseProfiles(localStorage.getItem(key) || '{}'));
            } catch {
                // One corrupt or inaccessible legacy source must not hide another valid source.
            }
            return profiles;
        }, Object.create(null));
    }

    function removeLegacyProfiles() {
        let failed = false;
        for (const key of legacyStorageKeys()) {
            try {
                localStorage.removeItem(key);
            } catch {
                failed = true;
            }
        }
        if (failed) throw new Error('legacy_profile_cleanup_failed');
    }

    function cleanupLegacyProfiles() {
        try {
            removeLegacyProfiles();
        } catch {
            createToast('Legacy profiles were saved, but a shared browser copy could not be removed.', 'error');
        }
    }

    function getProfiles() {
        try {
            const key = scopedStorageKey();
            if (!key) return Object.create(null);
            let profiles = parseProfiles(localStorage.getItem(key) || '{}');
            const legacyProfiles = getLegacyProfiles();
            if (
                resolvedOwnerKey() === 'direct_debug'
                && Object.keys(profiles).length === 0
                && Object.keys(legacyProfiles).length > 0
            ) {
                localStorage.setItem(key, JSON.stringify(legacyProfiles));
                profiles = legacyProfiles;
                cleanupLegacyProfiles();
            }
            return profiles;
        } catch {
            return Object.create(null);
        }
    }

    function writeProfiles(profiles) {
        try {
            const key = scopedStorageKey();
            if (!key) throw new Error('profile_owner_unavailable');
            localStorage.setItem(key, JSON.stringify(profiles));
            return true;
        } catch {
            createToast('Saved profiles could not be updated in browser storage.', 'error');
            announcePortalStatus('profile', 'Saved profiles could not be updated in browser storage.', { force: true });
            return false;
        }
    }

    function currentRecord() {
        return {
            pipeline: String(state.pipeline || 'lux-depth-v3'),
            config: copyDraftConfig()
        };
    }

    const baselineRecord = unprofiledBaselineRecord
        && typeof unprofiledBaselineRecord === 'object'
        && !Array.isArray(unprofiledBaselineRecord)
        ? unprofiledBaselineRecord
        : currentRecord();

    function recordsMatch(left, right) {
        if (!left || !right) return false;
        try {
            return JSON.stringify(left) === JSON.stringify(right);
        } catch {
            return false;
        }
    }

    function activeName() {
        return normalizeName(state.portalUi.activeProfileName || els.profileSelect?.value);
    }

    function setActiveName(name) {
        const normalized = normalizeName(name);
        state.portalUi.activeProfileName = normalized;
        if (els.profileSelect) els.profileSelect.value = normalized;
        return normalized;
    }

    function draftState(selectedNameOverride) {
        const selectedName = selectedNameOverride === undefined
            ? activeName()
            : normalizeName(selectedNameOverride);
        const profiles = getProfiles();
        const savedRecord = hasProfile(profiles, selectedName) ? profiles[selectedName] : null;
        const saved = Boolean(savedRecord && recordsMatch(savedRecord, currentRecord()));
        const dirty = Boolean(selectedName && savedRecord && !saved);
        const missingActiveProfile = Boolean(selectedName && !savedRecord);
        const unprofiledChanged = Boolean(
            !selectedName
            && (protectRestoredUnprofiledDraft || !recordsMatch(baselineRecord, currentRecord()))
        );
        return {
            activeName: selectedName,
            saved,
            dirty,
            missingActiveProfile,
            unprofiledChanged,
            hasProtectedChanges: dirty || missingActiveProfile || unprofiledChanged,
            unsaved: !selectedName || !savedRecord,
            profiles
        };
    }

    function syncDraftState() {
        const snapshot = draftState();
        if (els.profileSelect) {
            els.profileSelect.dataset.draftState = snapshot.saved ? 'saved' : snapshot.dirty ? 'dirty' : 'unsaved';
        }
        if (els.saveProfileBtn) {
            const label = snapshot.saved ? 'Saved' : snapshot.dirty ? 'Save Changes' : 'Save Profile';
            els.saveProfileBtn.textContent = label;
            els.saveProfileBtn.dataset.profileState = snapshot.saved ? 'saved' : snapshot.dirty ? 'dirty' : 'unsaved';
            els.saveProfileBtn.setAttribute(
                'aria-label',
                snapshot.saved
                    ? `Manage saved profile ${snapshot.activeName}. Current draft is saved.`
                    : snapshot.dirty
                        ? snapshot.activeName
                            ? `Manage saved profile ${snapshot.activeName}. Current draft has unsaved changes.`
                            : 'Manage saved profiles. Current unprofiled draft has unsaved changes.'
                        : snapshot.missingActiveProfile
                            ? `Manage saved profiles. ${snapshot.activeName} is no longer available; the current draft is unsaved.`
                        : snapshot.unprofiledChanged
                            ? 'Manage saved profiles. Current unprofiled draft has unsaved changes.'
                            : 'Manage saved profiles. Current draft is not saved.'
            );
        }
        if (isOpen()) renderDialog();
        return snapshot;
    }

    function refreshDropdown(preferredName = activeName()) {
        if (!els.profileSelect) return;
        const profiles = getProfiles();
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = 'Select Profile...';
        const options = Object.keys(profiles).sort().map((name) => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            return option;
        });
        els.profileSelect.replaceChildren(placeholder, ...options);
        setActiveName(Object.prototype.hasOwnProperty.call(profiles, preferredName) ? preferredName : '');
        syncDraftState();
    }

    function resetConfirmation() {
        if (!refs) return;
        refs.pendingAction = '';
        refs.pendingName = '';
    }

    function setMessage(message, tone = 'info') {
        if (!refs) return;
        refs.message.textContent = String(message || '');
        refs.message.dataset.tone = tone;
        refs.message.setAttribute('role', tone === 'error' ? 'alert' : 'status');
        refs.message.setAttribute('aria-live', tone === 'error' ? 'assertive' : 'polite');
    }

    function requireConfirmation(action, name, message, button, confirmLabel) {
        if (!refs) return false;
        if (refs.pendingAction === action && refs.pendingName === name) return true;
        refs.pendingAction = action;
        refs.pendingName = name;
        refs.validationVisible = false;
        renderDialog();
        setMessage(message, 'warning');
        button.textContent = confirmLabel;
        button.focus();
        return false;
    }

    function renderDialog() {
        if (!refs) return;
        const snapshot = draftState();
        const candidateName = normalizeName(refs.nameInput.value);
        const validationMessage = nameValidationMessage(candidateName);
        const showValidation = Boolean(validationMessage && refs.validationVisible);
        const candidateExists = hasProfile(snapshot.profiles, candidateName);
        const candidateMatchesDraft = candidateExists && recordsMatch(snapshot.profiles[candidateName], currentRecord());

        refs.state.textContent = snapshot.saved
            ? `Saved profile “${snapshot.activeName}” matches the current draft.`
            : snapshot.dirty
                ? snapshot.activeName
                    ? `Current draft has unsaved changes relative to “${snapshot.activeName}”.`
                    : 'Current draft has unsaved changes.'
                : snapshot.missingActiveProfile
                    ? `Saved profile “${snapshot.activeName}” is no longer available. The current draft remains unsaved.`
                : snapshot.unprofiledChanged
                    ? 'Current unprofiled draft has unsaved changes.'
                : 'Current draft is not saved to a profile.';
        refs.state.dataset.profileState = snapshot.saved ? 'saved' : snapshot.dirty ? 'dirty' : 'unsaved';
        refs.nameInput.setAttribute('aria-invalid', showValidation ? 'true' : 'false');
        if (showValidation) refs.nameInput.setAttribute('aria-errormessage', refs.message.id);
        else refs.nameInput.removeAttribute('aria-errormessage');

        const confirmingOverwrite = refs.pendingAction === 'overwrite' && refs.pendingName === candidateName;
        const confirmingRename = refs.pendingAction === 'rename-overwrite' && refs.pendingName === candidateName;
        const confirmingDelete = refs.pendingAction === 'delete' && refs.pendingName === snapshot.activeName;
        const legacyProfileCount = Object.keys(getLegacyProfiles()).length;
        const confirmingLegacyImport = refs.pendingAction === 'import-legacy';
        refs.pendingLoadButton.classList.toggle('hidden', !pendingLoadName);
        refs.pendingLoadButton.textContent = pendingLoadName
            ? `Discard Changes & Load “${pendingLoadName}”`
            : 'Discard Changes & Load';
        refs.importLegacyButton.classList.toggle('hidden', legacyProfileCount === 0);
        refs.importLegacyButton.textContent = confirmingLegacyImport
            ? 'Confirm Claim & Import'
            : `Import ${legacyProfileCount} Legacy Profile${legacyProfileCount === 1 ? '' : 's'}`;
        refs.saveButton.disabled = Boolean(validationMessage || candidateMatchesDraft);
        refs.saveButton.textContent = confirmingOverwrite
            ? 'Confirm Overwrite'
            : candidateExists ? 'Overwrite Profile' : 'Save New Profile';
        refs.renameButton.disabled = Boolean(
            validationMessage
            || !snapshot.activeName
            || !hasProfile(snapshot.profiles, snapshot.activeName)
            || candidateName === snapshot.activeName
        );
        refs.renameButton.textContent = confirmingRename ? 'Confirm Rename + Overwrite' : 'Rename Profile';
        refs.deleteButton.disabled = !hasProfile(snapshot.profiles, snapshot.activeName);
        refs.deleteButton.textContent = confirmingDelete ? 'Confirm Delete' : 'Delete Profile';
    }

    function ensureDialog() {
        if (refs) return refs;
        const overlay = document.createElement('div');
        overlay.id = 'profileManagerDialog';
        overlay.className = 'fixed inset-0 z-50 hidden items-center justify-center bg-slate-900/55 backdrop-blur-sm p-4 dark:bg-black/70';
        overlay.setAttribute('aria-hidden', 'true');
        overlay.dataset.ui = 'profile-manager-dialog';
        overlay.innerHTML = `
            <form class="w-full max-w-2xl max-h-[92vh] overflow-y-auto custom-scrollbar rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl dark:border-slate-800 dark:bg-slate-900" role="dialog" aria-modal="true" aria-labelledby="profileManagerTitle" aria-describedby="profileManagerState profileManagerMessage">
                <div class="flex items-start justify-between gap-4">
                    <div>
                        <p class="section-kicker">Local Profiles</p>
                        <h2 id="profileManagerTitle" class="mt-2 text-2xl font-extrabold tracking-tight text-slate-900 dark:text-white">Manage saved profile</h2>
                        <p id="profileManagerState" class="mt-2 text-[12px] leading-6 text-slate-600 dark:text-slate-300"></p>
                    </div>
                    <button type="button" data-profile-action="close" class="operator-action-btn operator-action-btn-secondary">Close</button>
                </div>
                <label class="mt-5 block text-xs font-medium text-slate-700 dark:text-slate-300" for="profileManagerName">Profile name</label>
                <input id="profileManagerName" type="text" maxlength="${PROFILE_NAME_MAX_LENGTH}" autocomplete="off" class="mt-2 w-full rounded-xl border border-slate-300 bg-slate-50 px-3 py-2.5 text-[13px] dark:border-slate-700 dark:bg-slate-800" aria-describedby="profileManagerMessage" />
                <p id="profileManagerMessage" class="field-status mt-2" role="status" aria-live="polite"></p>
                <button type="button" data-profile-action="load-pending" data-tone="blocked" class="operator-action-btn operator-action-btn-secondary hidden">Discard Changes &amp; Load</button>
                <button type="button" data-profile-action="import-legacy" class="operator-action-btn operator-action-btn-secondary hidden">Import Legacy Profiles</button>
                <div class="mt-5 flex flex-wrap items-center justify-between gap-3">
                    <button type="button" data-profile-action="delete" data-tone="blocked" class="operator-action-btn operator-action-btn-secondary">Delete Profile</button>
                    <div class="flex flex-wrap items-center gap-2">
                        <button type="button" data-profile-action="rename" class="operator-action-btn operator-action-btn-secondary">Rename Profile</button>
                        <button type="submit" data-profile-action="save" class="operator-action-btn operator-action-btn-primary">Save New Profile</button>
                    </div>
                </div>
            </form>`;
        document.body.appendChild(overlay);
        refs = {
            overlay,
            panel: overlay.querySelector('[role="dialog"]'),
            form: overlay.querySelector('form'),
            state: overlay.querySelector('#profileManagerState'),
            nameInput: overlay.querySelector('#profileManagerName'),
            message: overlay.querySelector('#profileManagerMessage'),
            pendingLoadButton: overlay.querySelector('[data-profile-action="load-pending"]'),
            importLegacyButton: overlay.querySelector('[data-profile-action="import-legacy"]'),
            closeButton: overlay.querySelector('[data-profile-action="close"]'),
            saveButton: overlay.querySelector('[data-profile-action="save"]'),
            renameButton: overlay.querySelector('[data-profile-action="rename"]'),
            deleteButton: overlay.querySelector('[data-profile-action="delete"]'),
            pendingAction: '',
            pendingName: '',
            validationVisible: false,
        };

        refs.nameInput.addEventListener('input', () => {
            pendingLoadName = '';
            resetConfirmation();
            refs.validationVisible = true;
            setMessage(nameValidationMessage(refs.nameInput.value), 'error');
            renderDialog();
        });
        refs.closeButton.addEventListener('click', close);
        overlay.addEventListener('click', (event) => {
            if (event.target === overlay) close();
        });
        refs.form.addEventListener('submit', (event) => {
            event.preventDefault();
            save();
        });
        refs.renameButton.addEventListener('click', rename);
        refs.deleteButton.addEventListener('click', remove);
        refs.pendingLoadButton.addEventListener('click', confirmPendingProfileLoad);
        refs.importLegacyButton.addEventListener('click', importLegacyProfiles);
        return refs;
    }

    function open(trigger = document.activeElement) {
        const dialog = ensureDialog();
        rememberOverlayTrigger(trigger);
        resetConfirmation();
        dialog.validationVisible = false;
        dialog.nameInput.value = activeName();
        setMessage('', 'info');
        renderDialog();
        dialog.overlay.classList.remove('hidden');
        dialog.overlay.classList.add('flex');
        dialog.overlay.setAttribute('aria-hidden', 'false');
        dialog.overlay.dataset.overlayOpen = 'true';
        setPortalBackgroundInert(true);
        dialog.nameInput.focus();
        dialog.nameInput.select();
    }

    function close() {
        if (!refs) return;
        refs.overlay.classList.add('hidden');
        refs.overlay.classList.remove('flex');
        refs.overlay.setAttribute('aria-hidden', 'true');
        refs.overlay.dataset.overlayOpen = 'false';
        pendingLoadName = '';
        resetConfirmation();
        setPortalBackgroundInert(false);
        restoreOverlayFocus();
    }

    function save() {
        const dialog = ensureDialog();
        const name = normalizeName(dialog.nameInput.value);
        const validationMessage = nameValidationMessage(name);
        if (validationMessage) {
            dialog.validationVisible = true;
            setMessage(validationMessage, 'error');
            renderDialog();
            dialog.nameInput.focus();
            return;
        }
        const profiles = getProfiles();
        const record = currentRecord();
        if (profiles[name] && !recordsMatch(profiles[name], record)) {
            if (!requireConfirmation(
                'overwrite',
                name,
                `Profile “${name}” already exists. Confirm overwrite to replace its saved configuration.`,
                dialog.saveButton,
                'Confirm Overwrite'
            )) return;
        }
        profiles[name] = record;
        if (!writeProfiles(profiles)) return;
        protectRestoredUnprofiledDraft = false;
        setActiveName(name);
        refreshDropdown(name);
        dialog.nameInput.value = name;
        dialog.validationVisible = false;
        resetConfirmation();
        setMessage(`Profile “${name}” saved.`, 'success');
        renderDialog();
        dialog.nameInput.focus({ preventScroll: true });
        createToast(`Profile “${name}” saved.`, 'success');
        announcePortalStatus('profile', `Profile ${name} saved.`, { force: true });
    }

    function rename() {
        const dialog = ensureDialog();
        const snapshot = draftState();
        const sourceName = snapshot.activeName;
        const nextName = normalizeName(dialog.nameInput.value);
        const validationMessage = nameValidationMessage(nextName);
        if (validationMessage) {
            dialog.validationVisible = true;
            setMessage(validationMessage, 'error');
            renderDialog();
            dialog.nameInput.focus();
            return;
        }
        if (!hasProfile(snapshot.profiles, sourceName)) {
            setMessage('Select a saved profile before renaming it.', 'error');
            return;
        }
        if (nextName === sourceName) {
            setMessage('Enter a different name to rename this profile.', 'error');
            return;
        }
        if (hasProfile(snapshot.profiles, nextName)) {
            if (!requireConfirmation(
                'rename-overwrite',
                nextName,
                `Profile “${nextName}” already exists. Confirm rename and overwrite to replace it.`,
                dialog.renameButton,
                'Confirm Rename + Overwrite'
            )) return;
        }
        const profiles = { ...snapshot.profiles, [nextName]: snapshot.profiles[sourceName] };
        delete profiles[sourceName];
        if (!writeProfiles(profiles)) return;
        setActiveName(nextName);
        refreshDropdown(nextName);
        dialog.nameInput.value = nextName;
        dialog.validationVisible = false;
        resetConfirmation();
        setMessage(`Profile “${sourceName}” renamed to “${nextName}”.`, 'success');
        renderDialog();
        dialog.nameInput.focus({ preventScroll: true });
        createToast(`Profile renamed to “${nextName}”.`, 'success');
        announcePortalStatus('profile', `Profile ${sourceName} renamed to ${nextName}.`, { force: true });
    }

    function remove() {
        const dialog = ensureDialog();
        const snapshot = draftState();
        const name = snapshot.activeName;
        if (!hasProfile(snapshot.profiles, name)) {
            setMessage('Select a saved profile before deleting it.', 'error');
            return;
        }
        if (!requireConfirmation(
            'delete',
            name,
            `Deleting “${name}” cannot be undone. The current draft will remain open.`,
            dialog.deleteButton,
            'Confirm Delete'
        )) return;
        const profiles = { ...snapshot.profiles };
        delete profiles[name];
        if (!writeProfiles(profiles)) return;
        setActiveName('');
        refreshDropdown('');
        dialog.nameInput.value = '';
        dialog.validationVisible = false;
        resetConfirmation();
        setMessage(`Profile “${name}” deleted. The current draft remains open and unsaved.`, 'success');
        renderDialog();
        dialog.nameInput.focus({ preventScroll: true });
        createToast(`Profile “${name}” deleted.`, 'success');
        announcePortalStatus('profile', `Profile ${name} deleted. The current draft remains open and unsaved.`, { force: true });
    }

    function applySelectedProfile(name) {
        if (!name) {
            protectRestoredUnprofiledDraft = false;
            setActiveName('');
            syncDraftState();
            return false;
        }
        const profiles = getProfiles();
        if (!hasProfile(profiles, name)) {
            const previousName = normalizeName(state.portalUi.activeProfileName);
            refreshDropdown(previousName);
            setMessage(`Profile “${name}” is no longer available in browser storage. The current draft was not changed.`, 'error');
            createToast(`Profile “${name}” is no longer available.`, 'error');
            announcePortalStatus('profile', `Profile ${name} is no longer available. The current draft was not changed.`, { force: true });
            return false;
        }
        protectRestoredUnprofiledDraft = false;
        setActiveName(name);
        state.pipeline = profiles[name].pipeline;
        state.config = JSON.parse(JSON.stringify(profiles[name].config));
        updateUIFromState();
        persistTransientDraft();
        void fetchPresetsForPipeline(state.pipeline, true);
        syncDraftState();
        createToast(`Profile “${name}” loaded.`);
        announcePortalStatus('profile', `Profile ${name} loaded.`, { force: true });
        return true;
    }

    function confirmPendingProfileLoad() {
        const name = pendingLoadName;
        if (!name) return;
        pendingLoadName = '';
        if (applySelectedProfile(name)) close();
    }

    function importLegacyProfiles() {
        const dialog = ensureDialog();
        const legacyProfiles = getLegacyProfiles();
        const legacyNames = Object.keys(legacyProfiles);
        if (legacyNames.length === 0) {
            resetConfirmation();
            setMessage('No legacy browser profiles are available to import.', 'info');
            renderDialog();
            return;
        }
        if (!requireConfirmation(
            'import-legacy',
            String(legacyNames.length),
            'Legacy profiles may have been shared by multiple portal accounts in this browser. Confirm to claim them for the current signed-in actor; existing actor-scoped profiles keep precedence.',
            dialog.importLegacyButton,
            'Confirm Claim & Import'
        )) return;

        const currentProfiles = getProfiles();
        const mergedProfiles = Object.assign(Object.create(null), legacyProfiles, currentProfiles);
        const importedCount = legacyNames.filter((name) => !hasProfile(currentProfiles, name)).length;
        if (!writeProfiles(mergedProfiles)) return;
        cleanupLegacyProfiles();
        refreshDropdown(activeName());
        resetConfirmation();
        setMessage(`${importedCount} legacy profile${importedCount === 1 ? '' : 's'} imported for the current actor.`, 'success');
        renderDialog();
        dialog.nameInput.focus({ preventScroll: true });
        announcePortalStatus(
            'profile',
            `${importedCount} legacy profile${importedCount === 1 ? '' : 's'} imported for the current actor.`,
            { force: true }
        );
    }

    function loadSelectedProfile(event) {
        const name = normalizeName(event?.target?.value);
        const previousName = normalizeName(state.portalUi.activeProfileName);
        const snapshot = draftState(previousName);
        if (name !== snapshot.activeName && snapshot.hasProtectedChanges) {
            if (event?.target) event.target.value = snapshot.activeName;
            if (!name) {
                open(event?.target || document.activeElement);
                setMessage(`Current draft has unsaved changes relative to “${snapshot.activeName}”. Save or discard those changes by choosing another saved profile before clearing the selection.`, 'warning');
                renderDialog();
                refs.nameInput.focus();
                return;
            }
            pendingLoadName = name;
            open(event?.target || document.activeElement);
            setMessage(`Current draft has unsaved changes relative to “${snapshot.activeName}”. Discard those changes only if you want to load “${name}”.`, 'warning');
            renderDialog();
            refs.pendingLoadButton.focus();
            return;
        }
        applySelectedProfile(name);
    }

    function isOpen() {
        return Boolean(refs && !refs.overlay.classList.contains('hidden'));
    }

    function activePanel() {
        return isOpen() ? refs.panel : null;
    }

    if (els.profileSelect && els.profileSelect.dataset.profileManagerBound !== 'true') {
        els.profileSelect.dataset.profileManagerBound = 'true';
        els.profileSelect.addEventListener('change', loadSelectedProfile);
    }
    refreshDropdown();

    return {
        activePanel,
        close,
        getProfiles,
        isOpen,
        open,
        refreshDropdown,
        syncDraftState,
    };
}
