'use client';
import { useState } from 'react';
import '../styles/PredictionResults.css';

export default function SpectrogramDisplay({ spectrogram }) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  if (!spectrogram) return null;

  return (
    <div className="spectrogram-display">
      <div class="flex justify-end">
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="px-4 py-2  text-sm font-medium bg-slate-200 rounded-lg hover:bg-slate-300 transition-colors"
        >
          {isCollapsed ? 'Show' : 'Hide'}
        </button>
      </div>
      <div className="flex justify-center items-center">
        <h2 className="spectrogram-display__title">Spectrogram</h2>
      </div>

      {!isCollapsed && (
        <div className="spectrogram-display__container">
          <img 
            src={spectrogram} 
            alt="Audio Spectrogram Visualization" 
            className="spectrogram-display__image" 
          />
          <div className="spectrogram-display__overlay">
            Visual representation of audio frequency content over time
          </div>
        </div>
      )}
    </div>
  );
}
