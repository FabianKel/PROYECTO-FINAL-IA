import React from "react";
import "../styles/FileSelector.css";

export default function FileSelector({ 
  defaultFiles, 
  selectedFile, 
  defaultFile, 
  onFileChange, 
  onDefaultFileChange, 
  onSubmit, 
  loading 
}) {
  return (
    <div className="file-selector">
      <div className="file-selector__header">
        <h2 className="file-selector__title">Audio Classifier</h2>
        <p className="file-selector__subtitle">Upload your audio file or select a default option</p>
      </div>
      
      <div className="file-selector__container">
        {/* Upload Section */}
        <div className={`file-selector__section ${selectedFile ? 'file-selector__section--valid' : ''}`}>
          <div className="file-selector__icon">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
            </svg>
          </div>
          <h3 className="file-selector__section-title">Upload Audio</h3>
          <input
            type="file"
            accept=".wav,.mp3,.m4a"
            onChange={onFileChange}
            className="file-selector__input"
          />
          {selectedFile && (
            <div className="file-selector__selected-file">
              ✓ {selectedFile.name}
            </div>
          )}
        </div>

        {/* Divider */}
        <div className="file-selector__divider">
          <div className="file-selector__divider-text">OR</div>
        </div>

        {/* Default Selection */}
        <div className={`file-selector__section ${defaultFile ? 'file-selector__section--valid' : ''}`}>
          <div className="file-selector__icon">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"/>
            </svg>
          </div>
          <h3 className="file-selector__section-title">Default Files</h3>
          <select
            value={defaultFile}
            onChange={onDefaultFileChange}
            className="file-selector__select"
          >
            <option value="">Choose a sample...</option>
            {defaultFiles.map(file => (
              <option key={file} value={file}>{file}</option>
            ))}
          </select>
          {defaultFile && (
            <div className="file-selector__selected-file">
              ✓ {defaultFile}
            </div>
          )}
        </div>
      </div>
      
      <div className="file-selector__button-container">
        <button
          onClick={onSubmit}
          disabled={loading || (!selectedFile && !defaultFile)}
          className={`file-selector__button ${loading ? 'file-selector__button--loading' : ''}`}
        >
          {loading ? (
            <>
              <span className="file-selector__spinner"></span>
              Processing...
            </>
          ) : (
            'Classify Audio'
          )}
        </button>
      </div>
    </div>
  );
}