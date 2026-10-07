import React from 'react';

export default function Navbar({ activeTab, setActiveTab, onSync, onRunInference, latency = 14 }) {
  return (
    <>
      {/* Top Minimalist Header */}
      <header className="sticky top-0 z-50 bg-white/90 backdrop-blur border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
          
          {/* Brand & Primary Nav */}
          <div className="flex items-center gap-8">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-sky-600 flex items-center justify-center text-white shadow-sm shadow-sky-600/20">
                <span className="material-symbols-outlined text-[20px]">speed</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg tracking-tight text-slate-900">RailPulse</span>
                <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">v4.2</span>
              </div>
            </div>

            {/* Clean Tabs */}
            <nav className="hidden md:flex items-center gap-1">
              <button
                onClick={() => setActiveTab('quick-check')}
                className={`px-3.5 py-1.5 rounded-lg text-sm font-semibold transition-colors ${
                  activeTab === 'quick-check'
                    ? 'bg-sky-50 text-sky-700'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                Quick Check
              </button>
              <button
                onClick={() => setActiveTab('batch-upload')}
                className={`px-3.5 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === 'batch-upload'
                    ? 'bg-sky-50 text-sky-700 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                Batch Analysis
              </button>
              <button
                onClick={() => setActiveTab('model-performance')}
                className={`px-3.5 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === 'model-performance'
                    ? 'bg-sky-50 text-sky-700 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                Model Drift
              </button>
              <button
                onClick={() => setActiveTab('quick-check')}
                className="px-3.5 py-1.5 rounded-lg text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors"
              >
                Fleet Nodes
              </button>
            </nav>
          </div>

          {/* Right Utility Badges & Profile */}
          <div className="flex items-center gap-3">
            {/* Live API Badge */}
            <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-full bg-slate-100 border border-slate-200 text-xs font-medium text-slate-600">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>FastAPI Active</span>
              <span className="text-slate-300">•</span>
              <span className="font-mono text-slate-500">{latency}ms</span>
            </div>

            {/* Division Select */}
            <div className="hidden lg:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-xs font-medium text-slate-700">
              <span className="material-symbols-outlined text-sm text-slate-400">train</span>
              <span>Bhusawal / Central</span>
            </div>

            <div className="h-4 w-px bg-slate-200 hidden sm:block"></div>

            {/* Notification Bell */}
            <button className="relative p-2 rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-900 transition-colors">
              <span className="material-symbols-outlined text-[20px]">notifications</span>
              <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-rose-500 rounded-full ring-2 ring-white"></span>
            </button>

            {/* User Avatar */}
            <div className="flex items-center gap-2 pl-1 cursor-pointer">
              <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-sky-600 to-sky-400 text-white font-semibold text-xs flex items-center justify-center shadow-sm">
                AV
              </div>
              <div className="hidden xl:flex flex-col text-left leading-none">
                <span className="text-xs font-semibold text-slate-900">Dr. A. Varma</span>
                <span className="text-[10px] text-slate-500 font-medium">Chief Rolling Stock Eng.</span>
              </div>
            </div>

          </div>
        </div>
      </header>

      {/* Sub-Bar / Operational Context */}
      <div className="border-b border-slate-200 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <span className="text-xs font-medium text-slate-400">Train Telemetry</span>
            <span className="text-slate-300">/</span>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-slate-900 text-sm">Trainset 20801</span>
              <span className="px-2 py-0.5 text-xs font-semibold rounded bg-sky-100 text-sky-800">VB-EXP V4</span>
              <span className="text-xs text-slate-500 font-mono hidden md:inline">Wardha Jn → Bhusawal</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 transition-all active:scale-95"
              onClick={onSync}
            >
              <span className="material-symbols-outlined text-sm text-slate-600" id="sync-icon">sync</span>
              Sync Telemetry
            </button>
            <button
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold rounded-lg bg-sky-600 hover:bg-sky-700 text-white shadow-sm shadow-sky-600/20 active:scale-95 transition-all"
              onClick={onRunInference}
            >
              <span className="material-symbols-outlined text-sm">bolt</span>
              Run Inference
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
