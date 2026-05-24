/* ═══════════════════════════════════════════════════════════
   WildLens AI — site.js
   ═══════════════════════════════════════════════════════════ */

/* ── DROP ZONE ───────────────────────────────────────────── */
function triggerFile() {
    document.getElementById('file-input').click();
}
function onFileSelected(input) {
    if (!input.files || !input.files[0]) return;
    showPreview(input.files[0]);
}
function showPreview(file) {
    const dz = document.getElementById('drop-zone');
    const img = document.getElementById('preview-img');
    if (!dz || !img) return;
    const reader = new FileReader();
    reader.onload = e => { img.src = e.target.result; dz.classList.add('has-img'); };
    reader.readAsDataURL(file);
    const btn = document.getElementById('btn-analyze');
    if (btn) btn.disabled = false;
}

document.addEventListener('DOMContentLoaded', () => {
    /* Drag & drop */
    const dz = document.getElementById('drop-zone');
    if (dz) {
        dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
        dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
        dz.addEventListener('drop', e => {
            e.preventDefault(); dz.classList.remove('drag');
            const file = e.dataTransfer?.files[0];
            if (file && file.type.startsWith('image/')) {
                const dt = new DataTransfer(); dt.items.add(file);
                const fi = document.getElementById('file-input');
                if (fi) fi.files = dt.files;
                showPreview(file);
            }
        });
    }

    /* Loading on submit */
    const form = document.getElementById('upload-form');
    if (form) {
        form.addEventListener('submit', () => {
            const ov = document.getElementById('loading-overlay');
            if (ov) ov.classList.add('active');
        });
    }

    /* Animate conf bars (det-list) */
    animateDetBars();

    /* Vẽ canvas nếu có DET_BOXES */
    if (typeof DET_BOXES !== 'undefined' && typeof DET_IMAGE !== 'undefined') {
        drawDetections(DET_IMAGE, DET_BOXES);
    }

    /* Stats chart */
    loadStatsChart();
});

/* ── ANIMATE DETECTION BARS ──────────────────────────────── */
function animateDetBars() {
    const fills = document.querySelectorAll('.det-conf-fill[data-width]');
    requestAnimationFrame(() => {
        fills.forEach(f => {
            const w = f.getAttribute('data-width');
            setTimeout(() => { f.style.width = w + '%'; }, 120);
        });
    });
    /* Legacy prob bars */
    document.querySelectorAll('.prob-fill[data-width]').forEach(f => {
        const w = f.getAttribute('data-width');
        setTimeout(() => { f.style.width = w + '%'; }, 120);
    });
}

/* ── DRAW BOUNDING BOXES ON CANVAS ───────────────────────── */
const DIET_COLORS = {
    herbivore: '#3ddc6b',
    carnivore: '#ff5252',
    omnivore: '#ffa726'
};
const DIET_LABEL_VI = {
    herbivore: 'Ăn cỏ',
    carnivore: 'Ăn thịt',
    omnivore: 'Tạp ăn'
};

function drawDetections(imgSrc, boxes) {
    const canvas = document.getElementById('det-canvas');
    if (!canvas || !boxes || boxes.length === 0) return;

    const img = new Image();
    img.onload = () => {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');

        /* Vẽ ảnh gốc */
        ctx.drawImage(img, 0, 0);

        /* Scale từ tọa độ normalized (0–1) sang pixel */
        const W = img.naturalWidth;
        const H = img.naturalHeight;

        boxes.forEach(box => {
            const color = DIET_COLORS[box.diet] || '#ffffff';
            const labelVI = DIET_LABEL_VI[box.diet] || box.diet;

            /* Tọa độ pixel */
            const px = box.x * W;
            const py = box.y * H;
            const pw = box.w * W;
            const ph = box.h * H;

            /* ── Vẽ khung ── */
            ctx.strokeStyle = color;
            ctx.lineWidth = Math.max(2, W * 0.003);
            ctx.strokeRect(px, py, pw, ph);

            /* ── Góc bo đẹp hơn ── */
            const r = Math.max(4, W * 0.005);
            drawRoundRect(ctx, px, py, pw, ph, r, color);

            /* ── Label background ── */
            const confStr = `${(box.conf * 100).toFixed(0)}%`;
            const labelStr = `${box.idx}  ${labelVI}  ${confStr}`;
            const fontSize = Math.max(11, W * 0.018);
            ctx.font = `600 ${fontSize}px 'Segoe UI', sans-serif`;
            const textW = ctx.measureText(labelStr).width;
            const padX = fontSize * 0.6;
            const padY = fontSize * 0.4;
            const boxH = fontSize + padY * 2;

            /* Bg pill */
            ctx.fillStyle = color + 'ee';
            const bgY = py > boxH + 4 ? py - boxH - 4 : py + 4;
            roundFill(ctx, px - 1, bgY, textW + padX * 2, boxH, 6);

            /* Text */
            ctx.fillStyle = '#000000';
            ctx.fillText(labelStr, px + padX - 1, bgY + boxH - padY - 1);
        });
    };
    img.src = imgSrc;
}

function drawRoundRect(ctx, x, y, w, h, r, color) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.closePath();
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2.5, ctx.canvas.width * 0.003);
    ctx.stroke();
}

function roundFill(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.closePath();
    ctx.fill();
}

/* ── TOAST ───────────────────────────────────────────────── */
function showToast(msg) {
    const t = document.getElementById('wlToast');
    const m = document.getElementById('wlToastMsg');
    if (!t || !m) return;
    m.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 3000);
}

/* ── INFO PAGE TABS ──────────────────────────────────────── */
function switchDiet(type) {
    document.querySelectorAll('.diet-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.info-content').forEach(c => c.classList.remove('active'));
    const tab = document.getElementById('tab-' + type);
    const content = document.getElementById('info-' + type);
    if (tab) tab.classList.add('active');
    if (content) content.classList.add('active');
}

/* ── STATS CHART ─────────────────────────────────────────── */
async function loadStatsChart() {
    const canvas = document.getElementById('dietPieChart');
    if (!canvas) return;
    try {
        const res = await fetch('/Home/StatsJson');
        const data = await res.json();
        if (data.total === 0) return;

        new Chart(canvas.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Ăn cỏ', 'Ăn thịt', 'Tạp ăn'],
                datasets: [{
                    data: [data.herbivore, data.carnivore, data.omnivore],
                    backgroundColor: ['#3ddc6b', '#ff5252', '#ffa726'],
                    borderColor: ['#2bb558', '#e03030', '#e09010'],
                    borderWidth: 2, hoverOffset: 8
                }]
            },
            options: {
                responsive: true, cutout: '60%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#7a9a7a', padding: 16, font: { size: 12 } }
                    }
                }
            }
        });

        const barCanvas = document.getElementById('dietBarChart');
        if (barCanvas) {
            new Chart(barCanvas.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: ['Ăn cỏ', 'Ăn thịt', 'Tạp ăn'],
                    datasets: [{
                        label: 'Số lần',
                        data: [data.herbivore, data.carnivore, data.omnivore],
                        backgroundColor: ['rgba(61,220,107,.3)', 'rgba(255,82,82,.3)', 'rgba(255,167,38,.3)'],
                        borderColor: ['#3ddc6b', '#ff5252', '#ffa726'],
                        borderWidth: 2, borderRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { color: '#7a9a7a' }, grid: { color: 'rgba(255,255,255,.05)' } },
                        y: { ticks: { color: '#7a9a7a', stepSize: 1 }, grid: { color: 'rgba(255,255,255,.05)' }, beginAtZero: true }
                    }
                }
            });
        }
    } catch (e) { console.warn('Chart load failed', e); }
}