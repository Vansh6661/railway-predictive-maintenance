/**
 * Indian Railways Predictive Maintenance System
 * Frontend Application Controller
 */

// Tab Navigation
document.querySelectorAll('.nav-tab').forEach(tabBtn => {
    tabBtn.addEventListener('click', () => {
        document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        
        tabBtn.classList.add('active');
        const targetId = tabBtn.getAttribute('data-tab');
        const targetPanel = document.getElementById(targetId);
        if (targetPanel) {
            targetPanel.classList.add('active');
        }
    });
});

// Update range value badge in real-time
function updateVal(fieldId) {
    const input = document.getElementById(fieldId);
    const badge = document.getElementById(`val-${fieldId}`);
    if (input && badge) {
        badge.innerText = input.value;
    }
    runPrediction();
}

// Preset Asset Configurations
let presets = {};

async function fetchPresets() {
    try {
        const res = await fetch('/api/presets');
        presets = await res.json();
    } catch (e) {
        console.error("Failed to load presets:", e);
    }
}

function loadPreset(presetKey) {
    document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
    
    // Highlight button
    if (presetKey === 'pristine_express') document.getElementById('btn-preset-pristine')?.classList.add('active');
    if (presetKey === 'monsoon_passenger') document.getElementById('btn-preset-monsoon')?.classList.add('active');
    if (presetKey === 'critical_freight') document.getElementById('btn-preset-critical')?.classList.add('active');
    
    const asset = presets[presetKey];
    if (!asset || !asset.data) return;
    
    const form = document.getElementById('telemetry-form');
    for (const [key, value] of Object.entries(asset.data)) {
        const input = form.elements[key];
        if (input) {
            input.value = value;
            const badge = document.getElementById(`val-${key}`);
            if (badge) badge.innerText = value;
        }
    }
    
    runPrediction();
}

// Debounced live prediction
let debounceTimer = null;

function runPrediction() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(executePrediction, 120);
}

async function executePrediction() {
    const form = document.getElementById('telemetry-form');
    if (!form) return;
    
    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => {
        payload[key] = value;
    });
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const result = await response.json();
        if (result.status === 'success') {
            updateDashboard(result);
        }
    } catch (err) {
        console.error("Prediction API Error:", err);
    }
}

function updateDashboard(data) {
    const pred = data.prediction;
    const subs = data.subsystems;
    
    // 1. Result Banner
    const banner = document.getElementById('prediction-banner');
    const title = document.getElementById('result-title');
    const icon = document.getElementById('result-icon');
    
    if (pred.maintenance_required) {
        banner.className = 'result-banner banner-danger';
        title.innerText = 'MAINTENANCE REQUIRED (DEFECT RISK)';
        icon.className = 'fa-solid fa-triangle-exclamation';
    } else {
        banner.className = 'result-banner banner-healthy';
        title.innerText = 'HEALTHY (NO MAINTENANCE REQUIRED)';
        icon.className = 'fa-solid fa-circle-check';
    }
    
    // 2. Risk Meter
    const riskPct = document.getElementById('risk-percentage');
    const progressBar = document.getElementById('risk-progress-bar');
    if (riskPct) riskPct.innerText = `${pred.rf_risk_probability}%`;
    if (progressBar) progressBar.style.width = `${Math.min(100, Math.max(0, pred.rf_risk_probability))}%`;
    
    // 3. Urgency & Window
    const urgBadge = document.getElementById('urgency-badge');
    const winRec = document.getElementById('window-recommendation');
    if (urgBadge) {
        urgBadge.innerText = pred.urgency;
        urgBadge.className = `badge-pill bg-${pred.urgency.toLowerCase()}`;
    }
    if (winRec) {
        winRec.innerText = pred.recommended_block_window;
    }
    
    // 4. Subsystems
    updateSubsystem('track', subs.track_health);
    updateSubsystem('mech', subs.mechanical_health);
    updateSubsystem('brake', subs.braking_health);
    updateSubsystem('elec', subs.electrical_health);
    
    // 5. Factors List
    const factorsList = document.getElementById('factors-list');
    if (factorsList) {
        factorsList.innerHTML = '';
        data.contributing_factors.forEach(factor => {
            const li = document.createElement('li');
            li.innerText = factor;
            factorsList.appendChild(li);
        });
    }
    
    // 6. Probabilities Check
    const lrProb = document.getElementById('lr-prob-val');
    const rfProb = document.getElementById('rf-prob-val');
    if (lrProb) lrProb.innerText = `${pred.lr_risk_probability}%`;
    if (rfProb) rfProb.innerText = `${pred.rf_risk_probability}%`;
}

function updateSubsystem(name, score) {
    const valElem = document.getElementById(`${name}-health-val`);
    const barElem = document.getElementById(`${name}-health-bar`);
    if (!valElem || !barElem) return;
    
    valElem.innerText = `${score}%`;
    barElem.style.width = `${score}%`;
    
    if (score >= 75) {
        barElem.className = 'sub-fill bg-success';
    } else if (score >= 50) {
        barElem.className = 'sub-fill bg-warning';
    } else {
        barElem.className = 'sub-fill bg-danger';
    }
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', async () => {
    await fetchPresets();
    executePrediction();
});
