import { useState, useEffect } from 'react';

// PredictionOptions.js

export default function PredictionOptions({ onOptionsChange }) {
  const [mode, setMode] = useState('full');  // default: 'full'
  const [useOffset, setUseOffset] = useState(false);

  useEffect(() => {
    const options = { mode };

    if (useOffset && (mode === 'full' || mode === 'single')) {
      options.offset = 5.0;
    }

    if (mode === 'single') {
      options.duration = 30.0;
    }

    if (mode === 'segment') {
      options.segmentDuration = 3.0;
    }

    onOptionsChange(options);
  }, [mode, useOffset]);

  return (
    <div className="mb-4">
      <h2 className="text-lg font-bold mb-2">Opciones de Predicción</h2>

      <div className="mb-2">
        <label className="block">
          <input
            type="radio"
            name="mode"
            value="full"
            checked={mode === 'full'}
            onChange={() => setMode('full')}
          />
          <span className="ml-2">Usar canción completa</span>
        </label>

        <label className="block">
          <input
            type="radio"
            name="mode"
            value="single"
            checked={mode === 'single'}
            onChange={() => setMode('single')}
          />
          <span className="ml-2">1 fragmento de 30 segundos</span>
        </label>

        <label className="block">
          <input
            type="radio"
            name="mode"
            value="segment"
            checked={mode === 'segment'}
            onChange={() => setMode('segment')}
          />
          <span className="ml-2">Varios fragmentos de 3 segundos</span>
        </label>
      </div>

      <div>
        <label>
          <input
            type="checkbox"
            checked={useOffset}
            disabled={mode === 'segment'}
            onChange={(e) => setUseOffset(e.target.checked)}
          />
          <span className="ml-2">Usar offset de 5 segundos</span>
        </label>
      </div>
    </div>
  );
}
