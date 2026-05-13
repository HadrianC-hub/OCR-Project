/* static/js/region_drawer.js
 *
 * Herramienta de dibujo de regiones de texto sobre el facsimilar
 * en la pantalla de edición.
 *
 * Sistema de coordenadas:
 *   - El <canvas> tiene atributos width/height iguales al tamaño
 *     NATURAL de la imagen (los píxeles del facsimilar original).
 *   - Su tamaño en CSS coincide con el tamaño de visualización del <img>.
 *   - Convertimos coords de cursor (CSS) → coords naturales para
 *     guardarlas/dibujarlas con un único factor (canvas.width / clientWidth).
 *
 * Modos:
 *   - "view"  : la herramienta no captura clicks (pasa al <img>).
 *   - "draw"  : click+arrastrar crea una nueva región. Click sobre una
 *               región existente la selecciona.
 *
 * Persistencia: cada vez que se modifica la lista (crear, borrar,
 * reordenar) hacemos POST al endpoint save_regions. El backend
 * normaliza order=1..N y nos devuelve la lista resultante.
 */

(function () {
  'use strict';

  const wrap = document.getElementById('facsDrawerWrap');
  if (!wrap) return;

  const img         = document.getElementById('facsImg');
  const canvas      = document.getElementById('facsCanvas');
  const ctx         = canvas.getContext('2d');
  const panel       = document.getElementById('regionsPanel');
  const list        = document.getElementById('regionList');
  const countSpan   = document.getElementById('regionCount');
  const clearBtn    = document.getElementById('clearRegionsBtn');
  const ocrBtn      = document.getElementById('ocrRegionsBtn');
  const modeGroup   = document.getElementById('modeBtnGroup');
  const zoomInBtn   = document.getElementById('zoomInBtn');
  const zoomOutBtn  = document.getElementById('zoomOutBtn');
  const zoomResetBtn= document.getElementById('zoomResetBtn');
  const textarea    = document.getElementById('pageTextarea');

  const saveUrl       = wrap.dataset.saveUrl;
  const ocrRegionsUrl = wrap.dataset.ocrRegionsUrl;

  let regions = [];
  try {
    regions = JSON.parse(wrap.dataset.initialRegions || '[]');
  } catch (e) { regions = []; }

  // ── State ──────────────────────────────────────────────────────────────
  let mode        = 'view';   // 'view' | 'draw'
  let scale       = 1;        // zoom multiplier (CSS scale)
  let selectedId  = null;     // selected region id
  let drawing     = null;     // {startX, startY, x, y, w, h} during drag
  let saveTimer   = null;     // debounce timer for saves

  // Palette identical to the segmentation viewer (consistency).
  const PALETTE = [
    '#2ecc71','#3498db','#e74c3c','#f1c40f','#9b59b6','#1abc9c','#e67e22',
  ];

  // ── Sizing ─────────────────────────────────────────────────────────────

  function setCanvasSize() {
    // Natural pixel dimensions of the source image
    const nw = img.naturalWidth  || img.width;
    const nh = img.naturalHeight || img.height;
    if (canvas.width  !== nw) canvas.width  = nw;
    if (canvas.height !== nh) canvas.height = nh;

    // CSS dimensions match the <img>'s rendered size.
    canvas.style.width  = img.clientWidth  + 'px';
    canvas.style.height = img.clientHeight + 'px';
  }

  function applyScale() {
    wrap.style.transform = 'scale(' + scale + ')';
    wrap.style.transformOrigin = 'top left';
  }

  // ── Drawing on canvas ──────────────────────────────────────────────────

  function colorFor(idx) { return PALETTE[idx % PALETTE.length]; }

  function redraw() {
    setCanvasSize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Existing regions
    regions.forEach((r, i) => {
      const isSel = (r.id === selectedId);
      const color = colorFor(i);

      ctx.strokeStyle = color;
      ctx.lineWidth   = isSel ? 6 : 4;
      ctx.fillStyle   = color + '22';   // ~13% opacity
      ctx.fillRect(r.x, r.y, r.width, r.height);
      ctx.strokeRect(r.x, r.y, r.width, r.height);

      // Label
      ctx.fillStyle = color;
      ctx.font = 'bold 28px sans-serif';
      const label = '#' + (r.order || (i + 1));
      ctx.fillText(label, r.x + 8, r.y + 32);
    });

    // Region in progress
    if (drawing) {
      ctx.strokeStyle = '#198754';
      ctx.lineWidth   = 4;
      ctx.setLineDash([10, 6]);
      ctx.fillStyle = '#19875422';
      ctx.fillRect(drawing.x, drawing.y, drawing.w, drawing.h);
      ctx.strokeRect(drawing.x, drawing.y, drawing.w, drawing.h);
      ctx.setLineDash([]);
    }
  }

  // ── Coordinate conversion ──────────────────────────────────────────────

  function cssToNatural(cssX, cssY) {
    // canvas.clientWidth is the *displayed* size; canvas.width is natural.
    // We must also undo the wrapper's CSS scale().
    const fx = canvas.width  / canvas.clientWidth;
    const fy = canvas.height / canvas.clientHeight;
    return { x: cssX * fx / scale, y: cssY * fy / scale };
  }

  function eventCanvasCoords(evt) {
    const rect = canvas.getBoundingClientRect();
    // rect.width includes the wrapper scale, so cssX/cssY are in
    // "displayed-after-scale" pixels — cssToNatural undoes both.
    const cssX = evt.clientX - rect.left;
    const cssY = evt.clientY - rect.top;
    return cssToNatural(cssX * (canvas.clientWidth / rect.width),
                        cssY * (canvas.clientHeight / rect.height));
  }

  // ── Mouse handlers (only active in 'draw' mode) ────────────────────────

  function onMouseDown(evt) {
    if (mode !== 'draw') return;
    if (evt.button !== 0) return;  // left click only

    const p = eventCanvasCoords(evt);
    const hit = hitRegion(p.x, p.y);
    if (hit) {
      selectedId = hit.id;
      renderList();
      redraw();
      return;
    }
    selectedId = null;
    drawing = { startX: p.x, startY: p.y, x: p.x, y: p.y, w: 0, h: 0 };
    redraw();
    evt.preventDefault();
  }

  function onMouseMove(evt) {
    if (!drawing) return;
    const p = eventCanvasCoords(evt);
    drawing.x = Math.min(drawing.startX, p.x);
    drawing.y = Math.min(drawing.startY, p.y);
    drawing.w = Math.abs(p.x - drawing.startX);
    drawing.h = Math.abs(p.y - drawing.startY);
    redraw();
  }

  function onMouseUp(evt) {
    if (!drawing) return;
    const w = Math.round(drawing.w), h = Math.round(drawing.h);
    if (w >= 8 && h >= 8) {
      const newR = {
        id:     'r' + Math.random().toString(36).slice(2, 10),
        order:  regions.length + 1,
        x:      Math.max(0, Math.round(drawing.x)),
        y:      Math.max(0, Math.round(drawing.y)),
        width:  w,
        height: h,
      };
      regions.push(newR);
      selectedId = newR.id;
      scheduleSave();
    }
    drawing = null;
    redraw();
    renderList();
  }

  function hitRegion(x, y) {
    // Search topmost (last drawn) first.
    for (let i = regions.length - 1; i >= 0; i--) {
      const r = regions[i];
      if (x >= r.x && x <= r.x + r.width &&
          y >= r.y && y <= r.y + r.height) {
        return r;
      }
    }
    return null;
  }

  // ── List rendering (right-side panel) ──────────────────────────────────

  function renderList() {
    if (!list) return;
    list.innerHTML = '';
    countSpan.textContent = regions.length;
    panel.style.display = regions.length ? '' : 'none';

    regions
      .slice()
      .sort((a, b) => (a.order || 0) - (b.order || 0))
      .forEach((r, i) => {
        const li = document.createElement('li');
        li.className = 'list-group-item d-flex align-items-center gap-2 region-item' +
                       (r.id === selectedId ? ' active-region' : '');
        li.draggable = true;
        li.dataset.regionId = r.id;
        li.innerHTML = `
          <span class="region-color-swatch" style="background:${colorFor(i)}"></span>
          <span class="region-handle text-muted" title="Arrastra para reordenar">⋮⋮</span>
          <span class="flex-grow-1 small">
            <strong>#${r.order || (i + 1)}</strong>
            <span class="text-muted">${r.width}×${r.height}px @ (${r.x}, ${r.y})</span>
          </span>
          <button type="button" class="btn btn-sm btn-outline-danger region-del-btn"
                  title="Eliminar región" data-region-id="${r.id}">
            <i class="bi bi-x"></i>
          </button>`;
        list.appendChild(li);
      });

    // Wire interactions
    list.querySelectorAll('.region-del-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteRegion(btn.dataset.regionId);
      });
    });
    list.querySelectorAll('.region-item').forEach((li) => {
      li.addEventListener('click', () => {
        selectedId = li.dataset.regionId;
        renderList();
        redraw();
      });
      // Drag-and-drop reorder
      li.addEventListener('dragstart', (e) => {
        li.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', li.dataset.regionId);
      });
      li.addEventListener('dragend', () => li.classList.remove('dragging'));
      li.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
      });
      li.addEventListener('drop', (e) => {
        e.preventDefault();
        const draggedId = e.dataTransfer.getData('text/plain');
        if (!draggedId || draggedId === li.dataset.regionId) return;
        reorderRegions(draggedId, li.dataset.regionId);
      });
    });
  }

  function deleteRegion(rid) {
    regions = regions.filter((r) => r.id !== rid);
    // Renumber order 1..N
    regions.sort((a, b) => (a.order || 0) - (b.order || 0));
    regions.forEach((r, i) => { r.order = i + 1; });
    if (selectedId === rid) selectedId = null;
    redraw();
    renderList();
    scheduleSave();
  }

  function reorderRegions(draggedId, targetId) {
    const dragged = regions.find((r) => r.id === draggedId);
    const target  = regions.find((r) => r.id === targetId);
    if (!dragged || !target) return;
    // Insert dragged immediately before target
    const sorted = regions.slice().sort((a, b) => (a.order || 0) - (b.order || 0));
    const rest   = sorted.filter((r) => r.id !== draggedId);
    const idx    = rest.indexOf(target);
    rest.splice(idx, 0, dragged);
    rest.forEach((r, i) => { r.order = i + 1; });
    regions = rest;
    redraw();
    renderList();
    scheduleSave();
  }

  // ── Persistence ────────────────────────────────────────────────────────

  function scheduleSave() {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(saveRegions, 500);
  }

  async function saveRegions() {
    try {
      const res = await fetch(saveUrl, {
        method:  'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': window.CSRF_TOKEN,
        },
        body: JSON.stringify({ regions: regions }),
      });
      if (!res.ok) {
        console.warn('save_regions returned', res.status);
        return;
      }
      const data = await res.json();
      if (Array.isArray(data.regions)) {
        regions = data.regions;
        renderList();
        redraw();
      }
    } catch (e) {
      console.error('Error saving regions:', e);
    }
  }

  // ── OCR over regions ───────────────────────────────────────────────────

  async function runOcrOnRegions() {
    if (!regions.length) return;
    if (!textarea) return;

    ocrBtn.disabled = true;
    const originalHTML = ocrBtn.innerHTML;
    ocrBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Procesando…';

    // Make sure the latest regions are persisted before OCR runs.
    await saveRegions();

    try {
      const res = await fetch(ocrRegionsUrl, {
        method:  'POST',
        headers: { 'X-CSRFToken': window.CSRF_TOKEN },
      });
      const data = await res.json();
      if (data.text !== undefined) {
        textarea.value = data.text;
        textarea.classList.add('border-success');
        setTimeout(() => textarea.classList.remove('border-success'), 2000);
      } else {
        alert('Error en OCR de regiones: ' + (data.error || 'desconocido'));
      }
    } catch (e) {
      alert('Error de conexión al ejecutar OCR de regiones.');
    } finally {
      ocrBtn.disabled = false;
      ocrBtn.innerHTML = originalHTML;
    }
  }

  // ── Mode handling ──────────────────────────────────────────────────────

  function setMode(newMode) {
    mode = newMode;
    if (modeGroup) {
      modeGroup.querySelectorAll('button').forEach((b) => {
        b.classList.toggle('active', b.dataset.mode === mode);
      });
    }
    canvas.style.pointerEvents = (mode === 'draw') ? 'auto' : 'none';
    canvas.style.cursor = (mode === 'draw') ? 'crosshair' : 'default';
  }

  // ── Wire-up ────────────────────────────────────────────────────────────

  function init() {
    setCanvasSize();
    setMode('view');
    redraw();
    renderList();
  }

  // Recompute canvas size when image finishes loading or window resizes
  if (img.complete && img.naturalWidth) {
    init();
  } else {
    img.addEventListener('load', init, { once: true });
  }
  window.addEventListener('resize', () => { setCanvasSize(); redraw(); });

  // Mouse / keyboard
  canvas.addEventListener('mousedown', onMouseDown);
  window.addEventListener('mousemove',  onMouseMove);
  window.addEventListener('mouseup',    onMouseUp);
  window.addEventListener('keydown', (e) => {
    if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId) {
      // Don't hijack delete inside the textarea.
      if (document.activeElement === textarea) return;
      deleteRegion(selectedId);
      e.preventDefault();
    }
    if (e.key === 'Escape') { selectedId = null; drawing = null; redraw(); renderList(); }
  });

  // Mode buttons
  if (modeGroup) {
    modeGroup.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-mode]');
      if (btn) setMode(btn.dataset.mode);
    });
  }

  // Zoom
  if (zoomInBtn)    zoomInBtn   .addEventListener('click', () => { scale = Math.min(4, +(scale + 0.25).toFixed(2)); applyScale(); });
  if (zoomOutBtn)   zoomOutBtn  .addEventListener('click', () => { scale = Math.max(0.5, +(scale - 0.25).toFixed(2)); applyScale(); });
  if (zoomResetBtn) zoomResetBtn.addEventListener('click', () => { scale = 1; applyScale(); });

  // List actions
  if (clearBtn) clearBtn.addEventListener('click', () => {
    if (!regions.length) return;
    if (!confirm('¿Eliminar todas las regiones de esta página?')) return;
    regions = []; selectedId = null;
    redraw(); renderList(); scheduleSave();
  });
  if (ocrBtn) ocrBtn.addEventListener('click', runOcrOnRegions);
})();
