import React, { useState, useEffect } from 'react';
import { API_ENDPOINTS } from '../../config';

export default function ModelPerformance() {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchModelInfo() {
      try {
        const res = await fetch(API_ENDPOINTS.MODEL_INFO);
        if (res.ok) {
          const data = await res.json();
          setModelInfo(data);
        }
      } catch (err) {
        console.error('Error fetching model info:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchModelInfo();
  }, []);

  if (loading || !modelInfo) {
    return (
      <div className="p-8 text-center text-xs text-slate-500">
        Loading model performance benchmarks and feature drift analytics...
      </div>
    );
  }

  const { comparison, top_features, dataset_summary, confusion_matrices } = modelInfo;

  return (
    <div className="space-y-6">
      
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-slate-900">Model Performance & Calibration Benchmarks</h2>
          <p className="text-xs text-slate-500 mt-0.5">Rigorous out-of-sample evaluation on <strong>20,000 unseen railway assets</strong> using stratified sampling.</p>
        </div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-50 border border-sky-200 text-sky-700 text-xs font-semibold">
          <span className="material-symbols-outlined text-sm">model_training</span>
          Random Forest vs Logistic Regression
        </div>
      </div>

      {/* Dataset Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <div className="bg-white p-4 rounded-xl border border-slate-200/90 shadow-sm">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Dataset Volume</div>
          <div className="text-2xl font-bold font-mono text-slate-900 mt-1">{dataset_summary.total_records.toLocaleString()}</div>
          <div className="text-xs text-slate-500 mt-0.5">80k Train / 20k Test Split</div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200/90 shadow-sm">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Normal Operating Assets</div>
          <div className="text-2xl font-bold font-mono text-emerald-600 mt-1">{dataset_summary.healthy_pct}%</div>
          <div className="text-xs text-slate-500 mt-0.5">{dataset_summary.healthy_count.toLocaleString()} Healthy Records</div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200/90 shadow-sm">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Defect Incidents (Target)</div>
          <div className="text-2xl font-bold font-mono text-rose-600 mt-1">{dataset_summary.maintenance_pct}%</div>
          <div className="text-xs text-slate-500 mt-0.5">{dataset_summary.maintenance_count.toLocaleString()} Maintenance Cases</div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200/90 shadow-sm">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Active Telemetry Dimensions</div>
          <div className="text-2xl font-bold font-mono text-sky-600 mt-1">{dataset_summary.features_count}</div>
          <div className="text-xs text-slate-500 mt-0.5">39 Encoded Matrix Features</div>
        </div>
      </div>

      {/* Model Benchmark Table */}
      <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-sky-600 text-[20px]">leaderboard</span>
            <h3 className="font-bold text-slate-900 text-sm">Classification Benchmark Table</h3>
          </div>
          <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-600">20,000 Test Records</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-400 uppercase font-mono text-[11px]">
                <th className="pb-3 font-semibold">Algorithm</th>
                <th className="pb-3 font-semibold">Accuracy</th>
                <th className="pb-3 font-semibold">Precision</th>
                <th className="pb-3 font-semibold">Recall</th>
                <th className="pb-3 font-semibold">F1-Score</th>
                <th className="pb-3 font-semibold">ROC-AUC</th>
                <th className="pb-3 font-semibold">Operational Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {comparison.map((m, idx) => {
                const isRF = m.model.includes('Random Forest');
                return (
                  <tr key={idx} className={isRF ? 'bg-sky-50/40 font-semibold' : ''}>
                    <td className="py-3.5 text-slate-900 font-bold">{m.model}</td>
                    <td className="py-3.5 font-mono text-slate-900">{m.accuracy}%</td>
                    <td className="py-3.5 font-mono text-emerald-600">{m.precision}%</td>
                    <td className="py-3.5 font-mono text-slate-700">{m.recall}%</td>
                    <td className="py-3.5 font-mono text-slate-700">{m.f1}%</td>
                    <td className="py-3.5 font-mono text-sky-600 font-bold">{m.roc_auc}%</td>
                    <td className="py-3.5">
                      <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                        isRF ? 'bg-sky-100 text-sky-800' : 'bg-slate-100 text-slate-600'
                      }`}>
                        {m.role}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature Importance & Confusion Matrix Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* Left: Feature Importance (7 cols) */}
        <div className="lg:col-span-7 bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-sky-600 text-[20px]">insights</span>
              <h3 className="font-bold text-slate-900 text-sm">Top Gini Feature Importances</h3>
            </div>
            <span className="text-xs text-slate-500">Random Forest Ensemble</span>
          </div>

          <div className="space-y-3 pt-2">
            {top_features.slice(0, 8).map((feat, idx) => (
              <div key={idx} className="space-y-1">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-800 font-mono text-[11px]">{feat.name} <span className="text-slate-400 font-sans">({feat.category})</span></span>
                  <span className="font-mono text-sky-600 font-semibold">{feat.importance}%</span>
                </div>
                <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-sky-600 rounded-full" style={{ width: `${(feat.importance / top_features[0].importance) * 100}%` }}></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Confusion Matrix (5 cols) */}
        <div className="lg:col-span-5 bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-sky-600 text-[20px]">grid_view</span>
              <h3 className="font-bold text-slate-900 text-sm">Confusion Matrix (Random Forest)</h3>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2.5 pt-2">
            <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 text-center">
              <div className="text-[10px] uppercase tracking-wider font-semibold text-emerald-700">True Negatives</div>
              <div className="text-xl font-bold font-mono text-emerald-800 mt-0.5">{confusion_matrices.random_forest.true_negatives.toLocaleString()}</div>
              <div className="text-[10px] text-emerald-600 mt-0.5">Correct Healthy</div>
            </div>

            <div className="p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-center">
              <div className="text-[10px] uppercase tracking-wider font-semibold text-rose-700">False Positives</div>
              <div className="text-xl font-bold font-mono text-rose-800 mt-0.5">{confusion_matrices.random_forest.false_positives.toLocaleString()}</div>
              <div className="text-[10px] text-rose-600 mt-0.5">False Alarms</div>
            </div>

            <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-200 text-center">
              <div className="text-[10px] uppercase tracking-wider font-semibold text-amber-700">False Negatives</div>
              <div className="text-xl font-bold font-mono text-amber-800 mt-0.5">{confusion_matrices.random_forest.false_negatives.toLocaleString()}</div>
              <div className="text-[10px] text-amber-600 mt-0.5">Missed Defects</div>
            </div>

            <div className="p-3.5 rounded-xl bg-sky-50 border border-sky-200 text-center">
              <div className="text-[10px] uppercase tracking-wider font-semibold text-sky-700">True Positives</div>
              <div className="text-xl font-bold font-mono text-sky-800 mt-0.5">{confusion_matrices.random_forest.true_positives.toLocaleString()}</div>
              <div className="text-[10px] text-sky-600 mt-0.5">Caught Defects</div>
            </div>
          </div>
          <p className="text-[11px] text-slate-500 pt-1">High precision (51.5%) minimizes unnecessary line stoppages across railway operations.</p>
        </div>

      </div>

    </div>
  );
}
