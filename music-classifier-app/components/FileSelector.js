export default function FileSelector({ defaultFiles, selectedFile, defaultFile, onFileChange, onDefaultFileChange, onSubmit, loading }) {
  return (
    <div className="mb-4">
      <h2 className="text-xl mb-2">Select Audio File</h2>
      <div className="flex gap-4">
        <div>
          <label className="block mb-1">Upload .wav file:</label>
          <input
            type="file"
            accept=".wav"
            onChange={onFileChange}
            className="border p-2"
          />
        </div>
        <div>
          <label className="block mb-1">Or select default file:</label>
          <select
            value={defaultFile}
            onChange={onDefaultFileChange}
            className="border p-2"
          >
            <option value="">Select a file</option>
            {defaultFiles.map(file => (
              <option key={file} value={file}>{file}</option>
            ))}
          </select>
        </div>
      </div>
      <button
        onClick={onSubmit}
        disabled={loading}
        className="mt-4 bg-blue-500 text-white p-2 rounded disabled:bg-gray-400"
      >
        {loading ? 'Processing...' : 'Classify Song'}
      </button>
    </div>
  );
}