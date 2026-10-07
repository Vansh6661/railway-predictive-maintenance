import React, { useState } from 'react';
import Navbar from './components/Navbar';
import QuickCheck from './components/QuickCheck/QuickCheck';
import BatchUpload from './components/BatchUpload/BatchUpload';
import ModelPerformance from './components/ModelPerformance/ModelPerformance';

export default function App() {
  const [activeTab, setActiveTab] = useState('quick-check');
  const [latency, setLatency] = useState(14);

  const handleSync = () => {
    const icon = document.getElementById('sync-icon');
    if (icon) {
      icon.classList.add('animate-spin');
      setTimeout(() => {
        icon.classList.remove('animate-spin');
        setLatency(Math.floor(12 + Math.random() * 5));
      }, 500);
    }
  };

  return (
    <div className="bg-slate-50 text-slate-900 min-h-screen flex flex-col font-sans">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onSync={handleSync}
        onRunInference={handleSync}
        latency={latency}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex-1 w-full">
        {activeTab === 'quick-check' && <QuickCheck onSyncComplete={() => setLatency(14)} />}
        {activeTab === 'batch-upload' && <BatchUpload />}
        {activeTab === 'model-performance' && <ModelPerformance />}
      </main>

      {/* Minimal Footer matching design */}
      <footer className="border-t border-slate-200 bg-white mt-auto py-4">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500">
          <div>RailPulse Telemetry Platform &bull; Ministry of Railways Mechanical Directorate</div>
          <div className="flex items-center gap-4 font-mono text-[11px] text-slate-400">
            <span>TLS 1.3 Calibrated</span>
            <span>NTP Sync: 0.2ms</span>
            <span>Node: BSL-DS-014</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
