import React, { useState, useRef } from 'react';
import { API_ENDPOINTS } from '../../config';

export default function BatchUpload() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [batchData, setBatchData] = useState(null);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [urgencyFilter, setUrgencyFilter] = useState('ALL');
  const [sortAsc, setSortAsc] = useState(false);

  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.csv')) {
        processUploadedFile(droppedFile);
      } else {
        setError('Please upload a valid .csv file.');
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processUploadedFile(e.target.files[0]);
    }
  };

  async function processUploadedFile(selectedFile) {
    setFile(selectedFile);
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(API_ENDPOINTS.PREDICT_BATCH, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.message || `Upload failed with status: ${res.status}`);
      }

      const data = await res.json();
      setBatchData(data);
    } catch (err) {
      console.error('Batch scoring error:', err);
      setError(err.message || 'Failed to process CSV file.');
      setBatchData(null);
    } finally {
      setLoading(false);
    }
  }

  async function downloadSample() {
    try {
      const res = await fetch(API_ENDPOINTS.SAMPLE_CSV);
      if (!res.ok) throw new Error('Failed to fetch template CSV');
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'sample_railway_fleet.csv';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Error downloading sample:', err);
      window.open(API_ENDPOINTS.SAMPLE_CSV, '_blank');
    }
  }

  function exportResultsCSV() {
    if (!batchData || !batchData.results) return;

    const headers = ['Asset ID', 'Verdict', 'Risk %', 'Urgency', 'Recommended Action', 'Primary Failure Risk'];
    const rows = batchData.results.map(r => [
      `"${r.asset_id || ''}"`,
      `"${r.label}"`,
      r.rf_risk_probability,
      `"${r.urgency}"`,
      `"${r.recommended_action}"`,
      `"${(r.top_contributing_features || []).join('; ')}"`
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `RailPulse_Fleet_Batch_Predictions_${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  const filteredResults = batchData ? batchData.results.filter(item => {
    const matchesSearch = !searchTerm || 
      (item.asset_id && item.asset_id.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (item.recommended_action && item.recommended_action.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (item.top_contributing_features && item.top_contributing_features.some(f => f.toLowerCase().includes(searchTerm.toLowerCase())));
    
    const matchesUrgency = urgencyFilter === 'ALL' || item.urgency === urgencyFilter;
    return matchesSearch && matchesUrgency;
  }).sort((a, b) => {
    return sortAsc ? a.rf_risk_probability - b.rf_risk_probability : b.rf_risk_probability - a.rf_risk_probability;
  }) : [];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-slate-900">Multi-Asset Fleet Batch Analysis</h2>
          <p className="text-xs text-slate-500 mt-0.5">Upload a fleet telemetry CSV dataset to run automated batch scoring and operational risk ranking.</p>
        </div>
        <button
          onClick={downloadSample}
          className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold rounded-lg bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 shadow-sm transition-all"
        >
          <span className="material-symbols-outlined text-sm text-sky-600">download</span>
          Download Fleet Template CSV
        </button>
      </div>

      {/* Upload Dropzone */}
      <div
        className={`bg-white rounded-2xl border-2 border-dashed p-8 text-center cursor-pointer transition-all flex flex-col items-center justify-center gap-3 shadow-sm ${
          isDragging ? 'border-sky-500 bg-sky-50/50' : 'border-slate-300 hover:border-slate-400'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current && fileInputRef.current.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={handleFileChange}
        />

        <div className="w-12 h-12 rounded-full bg-sky-50 text-sky-600 flex items-center justify-center">
          <span className="material-symbols-outlined text-[28px]">
            {loading ? 'hourglass_top' : 'cloud_upload'}
          </span>
        </div>

        <div>
          <h3 className="text-sm font-bold text-slate-900">
            {file ? `Selected Fleet File: ${file.name}` : 'Drag and drop Fleet CSV here, or click to browse'}
          </h3>
          <p className="text-xs text-slate-500 mt-1">Supports multi-asset telemetry, mechanical wear, and operational sensor CSV records.</p>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs font-medium flex items-center gap-2">
          <span className="material-symbols-outlined text-sm">warning</span>
          {error}
        </div>
      )}

      {/* Batch Results View */}
      {batchData && (
        <div className="space-y-5">
          
          {/* Summary KPI Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
            <div className="bg-white p-4 rounded-xl border border-slate-200/90 shadow-sm flex items-center gap-3.5">
              <div className="w-10 h-10 rounded-lg bg-sky-50 text-sky-600 flex items-center justify-center">
                <span className="material-symbols-outlined text-[20px]">dataset</span>
              </div>
              <div>
                <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Fleet Units Scored</div>
                <div className="text-xl font-bold font-mono text-slate-900 mt-0.5">{batchData.total_assets}</div>
              </div>
            </div>

            <div className="bg-white p-4 rounded-xl border border-slate-200/90 shadow-sm flex items-center gap-3.5">
              <div className="w-10 h-10 rounded-lg bg-rose-50 text-rose-600 flex items-center justify-center">
                <span className="material-symbols-outlined text-[20px]">warning</span>
              </div>
              <div>
                <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Critical Interventions</div>
                <div className="text-xl font-bold font-mono text-rose-600 mt-0.5">{batchData.critical_count}</div>
              </div>
            </div>

            <div className="bg-white p-4 rounded-xl border border-slate-200/90 shadow-sm flex items-center gap-3.5">
              <div className="w-10 h-10 rounded-lg bg-amber-50 text-amber-600 flex items-center justify-center">
                <span className="material-symbols-outlined text-[20px]">priority_high</span>
              </div>
              <div>
                <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Priority Attention</div>
                <div className="text-xl font-bold font-mono text-amber-600 mt-0.5">{batchData.high_count}</div>
              </div>
            </div>

            <div className="bg-white p-4 rounded-xl border border-slate-200/90 shadow-sm flex items-center gap-3.5">
              <div className="w-10 h-10 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
                <span className="material-symbols-outlined text-[20px]">verified</span>
              </div>
              <div>
                <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Healthy Fleet Units</div>
                <div className="text-xl font-bold font-mono text-emerald-600 mt-0.5">{batchData.low_count}</div>
              </div>
            </div>
          </div>

          {/* Ranked Table */}
          <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-4">
            
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 w-full sm:w-72">
                <span className="material-symbols-outlined text-sm text-slate-400">search</span>
                <input
                  type="text"
                  placeholder="Filter by Asset ID or symptom..."
                  value={searchTerm}
                  onChange={e => setSearchTerm(e.target.value)}
                  className="bg-transparent text-xs text-slate-900 outline-none w-full placeholder-slate-400"
                />
              </div>

              <div className="flex items-center gap-2 overflow-x-auto">
                {['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].map(lvl => (
                  <button
                    key={lvl}
                    onClick={() => setUrgencyFilter(lvl)}
                    className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                      urgencyFilter === lvl
                        ? 'bg-sky-600 text-white shadow-sm'
                        : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                    }`}
                  >
                    {lvl}
                  </button>
                ))}

                <button
                  onClick={exportResultsCSV}
                  className="inline-flex items-center gap-1 px-3 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-xs font-semibold text-slate-700 transition-all ml-2 whitespace-nowrap"
                >
                  <span className="material-symbols-outlined text-sm">file_download</span>
                  Export CSV
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-400 uppercase font-mono text-[11px]">
                    <th className="pb-3 font-semibold">Asset ID</th>
                    <th className="pb-3 font-semibold">Defect Risk</th>
                    <th className="pb-3 font-semibold">Urgency Tier</th>
                    <th className="pb-3 font-semibold">Verdict</th>
                    <th className="pb-3 font-semibold">Primary Telemetry Anomaly</th>
                    <th className="pb-3 font-semibold">Operational Directive</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {filteredResults.map((r, i) => {
                    const isCrit = r.urgency === 'CRITICAL';
                    const isHigh = r.urgency === 'HIGH';
                    const barColor = isCrit ? 'bg-rose-500' : (isHigh ? 'bg-amber-500' : 'bg-emerald-500');
                    const textColor = isCrit ? 'text-rose-600' : (isHigh ? 'text-amber-600' : 'text-emerald-600');

                    return (
                      <tr key={i} className="hover:bg-slate-50/80 transition-colors">
                        <td className="py-3 font-mono font-semibold text-slate-900">{r.asset_id}</td>
                        <td className="py-3">
                          <div className="flex items-center gap-2">
                            <div className="w-14 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                              <div className={`h-full rounded-full ${barColor}`} style={{ width: `${r.rf_risk_probability}%` }}></div>
                            </div>
                            <span className={`font-mono font-bold ${textColor}`}>{r.rf_risk_probability}%</span>
                          </div>
                        </td>
                        <td className="py-3">
                          <span className={`px-2 py-0.5 rounded text-[11px] font-semibold font-mono ${
                            isCrit ? 'bg-rose-50 text-rose-700 border border-rose-200' :
                            isHigh ? 'bg-amber-50 text-amber-700 border border-amber-200' :
                            'bg-emerald-50 text-emerald-700 border border-emerald-200'
                          }`}>
                            {r.urgency}
                          </span>
                        </td>
                        <td className="py-3">
                          <span className={`font-semibold ${r.maintenance_required ? 'text-rose-600' : 'text-emerald-600'}`}>
                            {r.maintenance_required ? 'Defect Action Required' : 'Nominal Safe'}
                          </span>
                        </td>
                        <td className="py-3 text-slate-600">
                          {(r.top_contributing_features || []).join(' • ')}
                        </td>
                        <td className="py-3 text-slate-500">{r.recommended_action}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

          </div>

        </div>
      )}

    </div>
  );
}
