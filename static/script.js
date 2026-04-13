/**
 * AgriLens — Elegant Client-Side Logic
 */

// ─────────────────────────────────────────────
//  DOM Elements
// ─────────────────────────────────────────────
const dropzone       = document.getElementById('dropzone');
const fileInput      = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const previewImg     = document.getElementById('preview-img');
const previewName    = document.getElementById('preview-filename');
const previewSize    = document.getElementById('preview-filesize');
const analyzeBtn     = document.getElementById('analyze-btn');
const clearBtn       = document.getElementById('clear-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const resultsSection = document.getElementById('results-section');
const toast          = document.getElementById('toast');
const toastMsg       = document.getElementById('toast-msg');
const themeToggle    = document.getElementById('theme-toggle');
const moonIcon       = document.getElementById('moon-icon');
const sunIcon        = document.getElementById('sun-icon');

let selectedFile = null;

// ─────────────────────────────────────────────
//  Theme Toggle Logic
// ─────────────────────────────────────────────
function initTheme() {
    const savedTheme = localStorage.getItem('agrilens-theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        document.body.classList.add('dark-mode');
        moonIcon.style.display = 'none';
        sunIcon.style.display = 'block';
    }
}

themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    
    if (isDark) {
        localStorage.setItem('agrilens-theme', 'dark');
        moonIcon.style.display = 'none';
        sunIcon.style.display = 'block';
    } else {
        localStorage.setItem('agrilens-theme', 'light');
        moonIcon.style.display = 'block';
        sunIcon.style.display = 'none';
    }
});

// Initialize Theme
initTheme();

// ─────────────────────────────────────────────
//  Drag & Drop Setup
// ─────────────────────────────────────────────
dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('drag-over');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

// ─────────────────────────────────────────────
//  File Handling
// ─────────────────────────────────────────────
function handleFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showToast('Please upload a valid image file (JPG, PNG, WebP)');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showToast('Image size must be under 10 MB');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewName.textContent = file.name;
        previewSize.textContent = formatFileSize(file.size);
        previewSection.classList.add('visible');
        dropzone.style.display = 'none';
        
        // Hide previous results if any
        resultsSection.classList.remove('visible');
    };
    reader.readAsDataURL(file);
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

clearBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewSection.classList.remove('visible');
    dropzone.style.display = 'block';
    resultsSection.classList.remove('visible');
});

// ─────────────────────────────────────────────
//  API Call
// ─────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showToast('Please select an image first');
        return;
    }

    showLoading(true);

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Diagnostic analysis failed');
        }

        if (data.success) {
            renderResults(data);
        } else {
            throw new Error(data.error || 'Unknown error');
        }
    } catch (error) {
        showToast(error.message || 'An error occurred during analysis.');
    } finally {
        showLoading(false);
    }
});

// ─────────────────────────────────────────────
//  Rendering Logic
// ─────────────────────────────────────────────
function renderResults(data) {
    const pred = data.prediction;
    const isDiseased = pred.status === 'DISEASED';

    // Left Column: Identity
    document.getElementById('result-img').src = previewImg.src;
    document.getElementById('result-plant').textContent = pred.plant;
    
    // Status Badge
    const statusBadge = document.getElementById('result-status');
    statusBadge.className = `status-badge ${isDiseased ? 'diseased' : 'healthy'}`;
    statusBadge.textContent = isDiseased ? 'Pathology Detected' : 'Optimal Health';

    // Confidence
    document.getElementById('confidence-value').textContent = pred.confidence.toFixed(1) + '%';

    // Top 3 Rendering
    const top3Container = document.getElementById('top3-list');
    top3Container.innerHTML = '';
    data.top3.forEach((item, index) => {
        if (index === 0) return; // Skip the main prediction
        const label = item.disease === 'None' ? 'Healthy' : item.disease;
        const div = document.createElement('div');
        div.className = 'top3-row';
        div.innerHTML = `
            <div class="top3-name">${label}</div>
            <div class="top3-bar-wrap">
                <div class="top3-bar-fill" style="width: 0%"></div>
            </div>
            <div class="top3-conf">${item.confidence.toFixed(1)}%</div>
        `;
        top3Container.appendChild(div);

        setTimeout(() => {
            div.querySelector('.top3-bar-fill').style.width = item.confidence + '%';
        }, 100 + index * 100);
    });

    // Right Column: Editorial Path
    document.getElementById('disease-title').textContent = isDiseased ? pred.disease : 'Health Diagnostic';
    document.getElementById('info-description').textContent = pred.description;
    document.getElementById('info-cause').textContent = pred.cause;
    document.getElementById('info-symptoms').textContent = pred.symptoms;
    document.getElementById('info-treatment').textContent = pred.treatment;
    document.getElementById('info-prevention').textContent = pred.prevention || 'Maintain current optimal agricultural practices.';

    // Severity Logic
    const severityLevels = ['None', 'Moderate', 'High', 'Critical'];
    const activeIndex = severityLevels.indexOf(pred.severity);
    
    document.getElementById('sev-mod').classList.toggle('active', activeIndex >= 1);
    document.getElementById('sev-high').classList.toggle('active', activeIndex >= 2);
    document.getElementById('sev-crit').classList.toggle('active', activeIndex >= 3);

    document.querySelectorAll('.severity-label').forEach((label) => {
        label.classList.toggle('active', label.dataset.level === pred.severity);
    });

    // Reveal UI
    resultsSection.classList.add('visible');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────
function showLoading(show) {
    loadingOverlay.classList.toggle('visible', show);
}

function showToast(message) {
    toastMsg.textContent = message;
    toast.classList.add('visible');
    setTimeout(() => {
        toast.classList.remove('visible');
    }, 4000);
}
