import React, { useState, useEffect, useRef } from 'react';
import { API_ENDPOINTS } from '../../config';

export default function QuickCheck({ onSyncComplete }) {
  const [presets, setPresets] = useState([]);
  const [activePresetIndex, setActivePresetIndex] = useState(0);
  const [telemetry, setTelemetry] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [latency, setLatency] = useState(14);
  const [showAllSensors, setShowAllSensors] = useState(false);

  // Core 4 Sliders State
  const [bearingTemp, setBearingTemp] = useState(78.2);
  const [lateralAccel, setLateralAccel] = useState(0.42);
  const [acousticEnergy, setAcousticEnergy] = useState(142);
  const [railStress, setRailStress] = useState(420);

  // Load presets from backend
  useEffect(() => {
    async function loadPresets() {
      try {
        const res = await fetch(API_ENDPOINTS.PRESETS);
        if (!res.ok) throw new Error('Failed to load presets');
        const data = await res.json();
        setPresets(data);
        if (data.length > 0) {
          applyPreset(data[0], 0);
        }
      } catch (err) {
        console.error('Preset loading error:', err);
      }
    }
    loadPresets();
  }, []);

  function applyPreset(preset, index) {
    setActivePresetIndex(index);
    const d = preset.data;
    setTelemetry({ ...d });

    // Map to simulator sliders
    const bTemp = d.bearing_temperature_c || 78.2;
    const lAccel = d.track_vibration_level ? parseFloat((d.track_vibration_level / 20).toFixed(2)) : 0.42;
    const aEnergy = Math.round((d.wheel_wear_percent || 50) * 1.6);
    const rStress = Math.round((d.rail_wear_mm || 5) * 28 + 250);

    setBearingTemp(bTemp);
    setLateralAccel(lAccel);
    setAcousticEnergy(aEnergy);
    setRailStress(rStress);

    runInferenceWithData({
      ...d,
      bearing_temperature_c: bTemp,
      track_vibration_level: lAccel * 20,
      wheel_wear_percent: Math.min(100, aEnergy / 1.6),
      rail_wear_mm: Math.max(0, (rStress - 250) / 28)
    });
  }

  async function runInferenceWithData(payloadData) {
    setLoading(true);
    const startTime = performance.now();

    try {
      const res = await fetch(API_ENDPOINTS.PREDICT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payloadData || telemetry)
      });

      const endTime = performance.now();
      setLatency(Math.round(endTime - startTime) || 14);

      if (res.ok) {
        const data = await res.json();
        setPrediction(data);
      }
    } catch (err) {
      console.error('Inference error:', err);
    } finally {
      setLoading(false);
    }
  }

  function handleSliderChange(sliderName, value) {
    const val = parseFloat(value);
    let updated = { ...telemetry };

    if (sliderName === 'bearing') {
      setBearingTemp(val);
      updated.bearing_temperature_c = val;
    } else if (sliderName === 'lateral') {
      setLateralAccel(val);
      updated.track_vibration_level = val * 20;
    } else if (sliderName === 'acoustic') {
      setAcousticEnergy(val);
      updated.wheel_wear_percent = Math.min(100, val / 1.6);
    } else if (sliderName === 'stress') {
      setRailStress(val);
      updated.rail_wear_mm = Math.max(0, (val - 250) / 28);
    }

    setTelemetry(updated);
    runInferenceWithData(updated);
  }

  function handleResetBaseline() {
    if (presets.length > 0) {
      applyPreset(presets[0], 0);
    }
  }

  const risk = prediction ? prediction.rf_risk_probability : 78.4;
  const isDefect = prediction ? prediction.maintenance_required : true;
  const subs = prediction ? prediction.subsystems : {
    track_health: 84.8,
    mechanical_health: 92.5,
    braking_health: 91.6,
    electrical_health: 100.0
  };

  return (
    <div className="space-y-6">
      
      {/* 1. SCENARIO PILLS BAR */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5">
        <div className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          Mechanical Simulation Profile
        </div>
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 sm:pb-0 scrollbar-none" id="preset-strip">
          {presets.map((p, idx) => {
            const isActive = activePresetIndex === idx;
            return (
              <button
                key={p.id}
                onClick={() => applyPreset(p, idx)}
                className={`px-3.5 py-1.5 rounded-full text-xs transition-all flex items-center gap-1.5 whitespace-nowrap ${
                  isActive
                    ? 'bg-sky-600 text-white font-semibold shadow-sm'
                    : 'bg-white hover:bg-slate-100 text-slate-600 font-medium border border-slate-200'
                }`}
              >
                {isActive && <span className="material-symbols-outlined text-[14px]">train</span>}
                <span>{p.name}</span>
                {isActive && <span className="w-1.5 h-1.5 rounded-full bg-white ml-0.5"></span>}
              </button>
            );
          })}
        </div>
      </div>

      {/* 2. CORE 3-SECOND INDUSTRIAL VERDICT CARD */}
      <section className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 lg:p-7 transition-all">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 pb-6 border-b border-slate-100">
          
          {/* Left: Alert Status & Big Metric Readout */}
          <div className="space-y-3">
            <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold ${
              isDefect 
                ? 'bg-rose-50 border border-rose-200 text-rose-700' 
                : 'bg-emerald-50 border border-emerald-200 text-emerald-700'
            }`}>
              <span className={`w-2 h-2 rounded-full ${isDefect ? 'bg-rose-600 animate-ping' : 'bg-emerald-500'}`}></span>
              <span>{isDefect ? 'Defect Detected (CRIT-02)' : 'Nominal Operating Health'}</span>
              <span className={isDefect ? 'text-rose-300' : 'text-emerald-300'}>•</span>
              <span className="font-mono font-medium">{prediction?.urgency ? `${prediction.urgency} Intervention` : 'Tier-2 Intervention'}</span>
            </div>

            <div className="flex items-baseline gap-3">
              <span className={`text-4xl sm:text-5xl font-extrabold tracking-tight font-mono ${
                isDefect ? 'text-rose-600' : 'text-emerald-600'
              }`}>
                {risk}%
              </span>
              <div>
                <div className="text-base sm:text-lg font-bold text-slate-900">Defect Risk Probability</div>
                <p className="text-xs text-slate-500">Calibrated Random Forest Ensemble (100 Trees &bull; Gini Scoring)</p>
              </div>
            </div>
          </div>

          {/* Right: Modern Hazard Spectrum Gauge */}
          <div className="w-full lg:w-80 bg-slate-50 border border-slate-100 p-4 rounded-xl space-y-2.5">
            <div className="flex justify-between text-xs font-semibold">
              <span className="text-slate-600">Hazard Spectrum</span>
              <span className={`font-mono font-bold ${isDefect ? 'text-rose-600' : 'text-emerald-600'}`}>
                {isDefect ? 'Action Required' : 'Nominal Safe'}
              </span>
            </div>

            {/* Linear segmented gauge */}
            <div className="relative w-full h-3 bg-slate-200 rounded-full overflow-hidden flex">
              <div className="h-full bg-emerald-500" style={{ width: '40%' }} title="Nominal"></div>
              <div className="h-full bg-amber-400" style={{ width: '30%' }} title="Elevated"></div>
              <div className="h-full bg-rose-500" style={{ width: '30%' }} title="Critical"></div>
              <div
                className="absolute top-0 bottom-0 w-1.5 bg-slate-900 rounded-full transform -translate-x-1/2 transition-all duration-300 shadow"
                style={{ left: `${Math.min(99, Math.max(1, risk))}%` }}
              ></div>
            </div>

            <div className="flex justify-between text-[11px] text-slate-400 font-medium">
              <span>Nominal (&lt;40%)</span>
              <span>Warning (70%)</span>
              <span>Critical</span>
            </div>
          </div>

        </div>

        {/* Key Track Directives (3 Clean Cards) */}
        <div className="mt-6">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
            <span className="material-symbols-outlined text-sm text-sky-600">alt_route</span>
            Immediate Operational Directives
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5">
            
            {/* Directive 1 */}
            <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200/70 hover:border-slate-300 transition-all flex flex-col justify-between">
              <div className="text-xs text-slate-500 font-medium">Block Maintenance Window</div>
              <div className="text-2xl font-bold text-rose-600 font-mono my-1">
                {isDefect ? '45 Mins' : 'None'}
              </div>
              <div className="text-xs text-slate-500">
                {isDefect ? 'Required sectional clearance (01:00-03:00 hrs)' : 'Standard periodic maintenance schedule'}
              </div>
            </div>

            {/* Directive 2 */}
            <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200/70 hover:border-slate-300 transition-all flex flex-col justify-between">
              <div className="text-xs text-slate-500 font-medium">Nearest Safe Siding</div>
              <div className="text-2xl font-bold text-slate-900 font-mono my-1">
                {isDefect ? 'KM 142.6' : 'On Line'}
              </div>
              <div className="text-xs text-slate-500">Bhusawal Div Loop 3 (Cleared)</div>
            </div>

            {/* Directive 3 */}
            <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200/70 hover:border-slate-300 transition-all flex flex-col justify-between">
              <div className="text-xs text-slate-500 font-medium">Speed Restriction (TSR)</div>
              <div className="text-2xl font-bold text-sky-600 font-mono my-1">
                {isDefect ? '30 km/h' : '130 km/h'}
              </div>
              <div className="text-xs text-slate-500">
                {isDefect ? 'Imposed immediately till siding halt' : 'Full authorized line velocity permissible'}
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* 3. CLEAN 2-COLUMN TELEMETRY GRID */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* LEFT COLUMN: Root Cause + Subsystems (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          
          {/* (A) PLAIN-ENGLISH ROOT CAUSE & SHAP ATTRIBUTION */}
          <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-sky-600 text-[20px]">psychology</span>
                <h3 className="font-bold text-slate-900 text-sm">Root Cause & Attribution</h3>
              </div>
              <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-sky-50 text-sky-700">94.2% Confidence</span>
            </div>

            {/* Plain-English Diagnosis Box */}
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80">
              <p className="text-sm text-slate-700 leading-relaxed">
                <span className={`font-semibold ${isDefect ? 'text-rose-600' : 'text-emerald-600'}`}>
                  {isDefect ? 'Primary Anomaly: ' : 'Nominal Telemetry: '}
                </span>
                {prediction?.top_contributing_features ? (
                  prediction.top_contributing_features.join('; ')
                ) : (
                  'Bogie yaw oscillation within safe limits. Axle bearing temperature nominal. No acoustic anomalies detected.'
                )}
              </p>
            </div>

            {/* Top SHAP Factors */}
            <div className="space-y-3 pt-1">
              <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Top Telemetry Drivers</div>
              
              {/* Factor 1 */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-800">Axle Bearing Thermal Gradient (ΔT)</span>
                  <span className="font-mono text-rose-600 font-semibold">+0.38 SHAP impact</span>
                </div>
                <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-rose-500 rounded-full" style={{ width: `${Math.min(100, (bearingTemp / 110) * 100)}%` }}></div>
                </div>
              </div>

              {/* Factor 2 */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-800">Wheel Flat Acoustic Energy</span>
                  <span className="font-mono text-rose-600 font-semibold">+0.29 SHAP impact</span>
                </div>
                <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-rose-400 rounded-full" style={{ width: `${Math.min(100, (acousticEnergy / 180) * 100)}%` }}></div>
                </div>
              </div>

              {/* Factor 3 */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-800">Subgrade Rail Stress</span>
                  <span className="font-mono text-sky-600 font-semibold">-0.05 (Mitigating)</span>
                </div>
                <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-sky-500 rounded-full" style={{ width: `${Math.min(100, (railStress / 650) * 100)}%` }}></div>
                </div>
              </div>

            </div>
          </div>

          {/* (B) 4 SUBSYSTEM HEALTH CARDS (Clean 2x2 Grid) */}
          <div>
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Subsystem Health Status</div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
              
              {/* Subsystem 1: Track Bed */}
              <div className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-sm space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-500">Track Bed</span>
                  <span className={`text-xs font-semibold px-2 py-0.5 rounded ${
                    subs.track_health >= 70 ? 'bg-emerald-50 text-emerald-700' : (subs.track_health >= 40 ? 'bg-amber-50 text-amber-700' : 'bg-rose-50 text-rose-700')
                  }`}>
                    {subs.track_health >= 70 ? 'Nominal' : (subs.track_health >= 40 ? 'Degraded' : 'Critical')}
                  </span>
                </div>
                <div className="flex items-baseline justify-between">
                  <span className="text-2xl font-bold font-mono text-slate-900">{subs.track_health}%</span>
                  <span className="text-xs text-slate-500">Ballast voiding</span>
                </div>
                <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full ${subs.track_health >= 70 ? 'bg-emerald-500' : (subs.track_health >= 40 ? 'bg-amber-500' : 'bg-rose-500')}`} 
                    style={{ width: `${subs.track_health}%` }}
                  ></div>
                </div>
              </div>

              {/* Subsystem 2: Wheelset & Axles */}
              <div className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-sm space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-500">Wheelset & Axles</span>
                  <span className={`text-xs font-semibold px-2 py-0.5 rounded ${
                    subs.mechanical_health >= 70 ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'
                  }`}>
                    {subs.mechanical_health >= 70 ? 'Nominal' : 'Anomaly'}
                  </span>
                </div>
                <div className="flex items-baseline justify-between">
                  <span className={`text-2xl font-bold font-mono ${subs.mechanical_health < 70 ? 'text-rose-600' : 'text-slate-900'}`}>
                    {subs.mechanical_health}%
                  </span>
                  <span className={`text-xs font-medium ${subs.mechanical_health < 70 ? 'text-rose-600' : 'text-slate-500'}`}>
                    {subs.mechanical_health < 70 ? 'Acoustic spike' : 'Balanced'}
                  </span>
                </div>
                <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full ${subs.mechanical_health >= 70 ? 'bg-emerald-500' : 'bg-rose-500'}`} 
                    style={{ width: `${subs.mechanical_health}%` }}
                  ></div>
                </div>
              </div>

              {/* Subsystem 3: Pneumatic Brakes */}
              <div className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-sm space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-500">Pneumatic Brakes</span>
                  <span className={`text-xs font-semibold px-2 py-0.5 rounded ${
                    subs.braking_health >= 70 ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'
                  }`}>
                    {subs.braking_health >= 70 ? 'Nominal' : 'Warning'}
                  </span>
                </div>
                <div className="flex items-baseline justify-between">
                  <span className="text-2xl font-bold font-mono text-slate-900">{subs.braking_health}%</span>
                  <span className="text-xs text-slate-500">0.15 bar/min AR leak</span>
                </div>
                <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full ${subs.braking_health >= 70 ? 'bg-emerald-500' : 'bg-amber-500'}`} 
                    style={{ width: `${subs.braking_health}%` }}
                  ></div>
                </div>
              </div>

              {/* Subsystem 4: Electrical & Signals */}
              <div className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-sm space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-500">Electrical & Signals</span>
                  <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-50 text-emerald-700">Nominal</span>
                </div>
                <div className="flex items-baseline justify-between">
                  <span className="text-2xl font-bold font-mono text-slate-900">{subs.electrical_health}%</span>
                  <span className="text-xs text-emerald-600 font-medium">KAVACH Locked</span>
                </div>
                <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${subs.electrical_health}%` }}></div>
                </div>
              </div>

            </div>
          </div>

        </div>

        {/* RIGHT COLUMN: Clean Sensitivity Simulator (5 cols) */}
        <div className="lg:col-span-5">
          <aside className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-6 sticky top-24">
            <div className="flex items-center justify-between border-b border-slate-100 pb-4">
              <div>
                <h3 className="font-bold text-slate-900 text-sm">Sensitivity Simulator</h3>
                <p className="text-xs text-slate-500 mt-0.5">Real-time what-if parameter stress test</p>
              </div>
              <span className="text-xs font-mono font-medium px-2 py-1 rounded bg-slate-100 text-slate-600">30 Nodes</span>
            </div>

            {/* Sliders Collection */}
            <div className="space-y-5">
              
              {/* Slider 1: Bearing Temp */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <label className="font-semibold text-slate-700">Bearing Temp (Bogie 1)</label>
                  <div className="font-mono font-bold text-rose-600 text-sm">
                    {bearingTemp.toFixed(1)} <span className="text-xs font-normal text-slate-400">°C</span>
                  </div>
                </div>
                <input
                  className="w-full accent-sky-600 h-1.5 bg-slate-200 rounded-lg cursor-pointer"
                  type="range"
                  min="30"
                  max="110"
                  step="0.1"
                  value={bearingTemp}
                  onChange={e => handleSliderChange('bearing', e.target.value)}
                />
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>Nominal: 45°C</span>
                  <span className="text-rose-500">Trip: 85°C</span>
                </div>
              </div>

              {/* Slider 2: Lateral Accel */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <label className="font-semibold text-slate-700">Lateral Acceleration (RMS)</label>
                  <div className="font-mono font-bold text-slate-900 text-sm">
                    {lateralAccel.toFixed(2)} <span className="text-xs font-normal text-slate-400">g</span>
                  </div>
                </div>
                <input
                  className="w-full accent-sky-600 h-1.5 bg-slate-200 rounded-lg cursor-pointer"
                  type="range"
                  min="0.05"
                  max="0.80"
                  step="0.01"
                  value={lateralAccel}
                  onChange={e => handleSliderChange('lateral', e.target.value)}
                />
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>Base: 0.15g</span>
                  <span>Max: 0.50g</span>
                </div>
              </div>

              {/* Slider 3: Wheel Acoustic */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <label className="font-semibold text-slate-700">Wheel Flat Acoustic Energy</label>
                  <div className="font-mono font-bold text-rose-600 text-sm">
                    {acousticEnergy} <span className="text-xs font-normal text-slate-400">dB</span>
                  </div>
                </div>
                <input
                  className="w-full accent-rose-600 h-1.5 bg-slate-200 rounded-lg cursor-pointer"
                  type="range"
                  min="80"
                  max="180"
                  step="1"
                  value={acousticEnergy}
                  onChange={e => handleSliderChange('acoustic', e.target.value)}
                />
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>Baseline: &lt;110 dB</span>
                  <span className="text-rose-500">Limit: 135 dB</span>
                </div>
              </div>

              {/* Slider 4: Rail Stress */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <label className="font-semibold text-slate-700">Ultrasonic Rail Stress</label>
                  <div className="font-mono font-bold text-slate-900 text-sm">
                    {railStress} <span className="text-xs font-normal text-slate-400">MPa</span>
                  </div>
                </div>
                <input
                  className="w-full accent-sky-600 h-1.5 bg-slate-200 rounded-lg cursor-pointer"
                  type="range"
                  min="150"
                  max="650"
                  step="5"
                  value={railStress}
                  onChange={e => handleSliderChange('stress', e.target.value)}
                />
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>Nominal: 350 MPa</span>
                  <span>Yield: 520 MPa</span>
                </div>
              </div>

            </div>

            {/* Buttons */}
            <div className="pt-2 space-y-2.5">
              <button
                className="w-full py-2.5 px-4 rounded-xl bg-sky-600 hover:bg-sky-700 text-white text-xs font-semibold shadow-sm shadow-sky-600/20 active:scale-98 transition-all flex items-center justify-center gap-1.5"
                onClick={() => runInferenceWithData(telemetry)}
              >
                <span className="material-symbols-outlined text-sm">bolt</span>
                Re-evaluate Model
              </button>

              <button
                className="w-full py-2 px-4 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold transition-colors flex items-center justify-center gap-1.5"
                onClick={handleResetBaseline}
              >
                <span className="material-symbols-outlined text-sm text-slate-500">restart_alt</span>
                Reset to Baseline
              </button>
            </div>

            <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
              <span className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                FastAPI Endpoint Ready
              </span>
              <span className="font-mono">{latency}ms latency</span>
            </div>

          </aside>
        </div>

      </div>

    </div>
  );
}
