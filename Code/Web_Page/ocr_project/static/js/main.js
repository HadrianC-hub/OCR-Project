/* ═══════════════════════════════════════════════
   ArchivoOCR — main.js
   ═══════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

  // ── Bootstrap tooltips ──────────────────────────────────────────────
  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
    new bootstrap.Tooltip(el, { trigger: 'hover', placement: 'bottom' });
  });

  // ── Auto-dismiss flash messages after 4 s ───────────────────────────
  document.querySelectorAll('#flash-messages .alert').forEach(alert => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
      if (bsAlert) bsAlert.close();
    }, 4000);
  });

  // ── Download modal — compute URL at click time (no race conditions) ─
  const dlModal = document.getElementById('dlModal');
  if (dlModal) {
    let currentDocId = null;

    const computeUrl = (format) => {
      const facsRadio = dlModal.querySelector('input[name="dlFacsimile"]:checked');
      const useFacs = facsRadio && facsRadio.value === '1';
      const facsParam = useFacs ? '&facsimile=1' : '';
      return `/documents/${currentDocId}/download/?format=${format}${facsParam}`;
    };

    dlModal.addEventListener('show.bs.modal', event => {
      const trigger  = event.relatedTarget;
      currentDocId   = trigger?.dataset?.docId;
      const docTitle = trigger?.dataset?.docTitle || 'este documento';

      const titleEl = dlModal.querySelector('#dlModalDocTitle');
      if (titleEl) titleEl.textContent = docTitle;

      // Reset to "edición de lectura"
      const modeText = dlModal.querySelector('#dlModeText');
      if (modeText) modeText.checked = true;
    });

    // Intercept the click on each download button and navigate at THAT
    // moment, reading the radio state fresh. Avoids any race condition
    // with the radio's change event.
    const pdfBtn  = dlModal.querySelector('#dlPdfBtn');
    const epubBtn = dlModal.querySelector('#dlEpubBtn');
    if (pdfBtn) {
      pdfBtn.addEventListener('click', e => {
        e.preventDefault();
        if (currentDocId) window.location.href = computeUrl('pdf');
      });
    }
    if (epubBtn) {
      epubBtn.addEventListener('click', e => {
        e.preventDefault();
        if (currentDocId) window.location.href = computeUrl('epub');
      });
    }
  }

  // ── Confirm dangerous actions (extra safeguard) ──────────────────────
  // Any form with data-confirm attribute gets a JS confirm dialog
  document.querySelectorAll('form[data-confirm]').forEach(form => {
    form.addEventListener('submit', e => {
      if (!window.confirm(form.dataset.confirm)) {
        e.preventDefault();
      }
    });
  });

  // ── Keep active nav icon highlighted ────────────────────────────────
  const path = window.location.pathname;
  document.querySelectorAll('#main-navbar a.nav-icon-btn').forEach(link => {
    if (link.getAttribute('href') && path.startsWith(link.getAttribute('href')) && link.getAttribute('href') !== '/') {
      link.classList.add('active', 'bg-primary', 'text-white', 'border-primary');
    }
  });

});
