import { useState, useEffect } from 'react';

export default function AudioPredictionSettings({ onOptionsChange }) {
  const [mode, setMode] = useState('full');
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
    <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-4 mb-4 max-w-lg mx-auto">
      <h2 className="text-lg font-semibold text-gray-800 mb-4 text-center">
        Opciones de Predicción
      </h2>

      <div className="space-y-2 mb-4">
        <label className="flex items-center p-2 bg-gray-50 rounded border-2 border-transparent hover:border-blue-200 transition-all cursor-pointer group">
          <input
            type="radio"
            name="mode"
            value="full"
            checked={mode === 'full'}
            onChange={() => setMode('full')}
            className="w-4 h-4 text-blue-500 border-gray-300 focus:ring-blue-500 focus:ring-1"
          />
          <span className="ml-2 text-sm text-gray-700 group-hover:text-blue-600 transition-colors">
            Usar canción completa
          </span>
        </label>

        <label className="flex items-center p-2 bg-gray-50 rounded border-2 border-transparent hover:border-blue-200 transition-all cursor-pointer group">
          <input
            type="radio"
            name="mode"
            value="single"
            checked={mode === 'single'}
            onChange={() => setMode('single')}
            className="w-4 h-4 text-blue-500 border-gray-300 focus:ring-blue-500 focus:ring-1"
          />
          <span className="ml-2 text-sm text-gray-700 group-hover:text-blue-600 transition-colors">
            1 fragmento de 30 segundos
          </span>
        </label>

        <label className="flex items-center p-2 bg-gray-50 rounded border-2 border-transparent hover:border-blue-200 transition-all cursor-pointer group">
          <input
            type="radio"
            name="mode"
            value="segment"
            checked={mode === 'segment'}
            onChange={() => setMode('segment')}
            className="w-4 h-4 text-blue-500 border-gray-300 focus:ring-blue-500 focus:ring-1"
          />
          <span className="ml-2 text-sm text-gray-700 group-hover:text-blue-600 transition-colors">
            Varios fragmentos de 3 segundos
          </span>
        </label>
      </div>

      <div className="border-t border-gray-200 pt-3">
        <label className="flex items-center p-2 bg-gray-50 rounded border-2 border-transparent hover:border-blue-200 transition-all cursor-pointer group">
          <input
            type="checkbox"
            checked={useOffset}
            disabled={mode === 'segment'}
            onChange={(e) => setUseOffset(e.target.checked)}
            className="w-4 h-4 text-blue-500 border-gray-300 focus:ring-blue-500 focus:ring-1 disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <span className={`ml-2 text-sm transition-colors ${
            mode === 'segment' 
              ? 'text-gray-400 cursor-not-allowed' 
              : 'text-gray-700 group-hover:text-blue-600'
          }`}>
            Usar offset de 5 segundos
          </span>
        </label>
        {mode === 'segment' && (
          <p className="text-xs text-gray-500 mt-1 ml-6">
            El offset no está disponible para fragmentos múltiples
          </p>
        )}
      </div>


    </div>
  );
}