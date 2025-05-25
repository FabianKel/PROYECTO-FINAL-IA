import '../styles/PredictionResults.css';
export default function SpectrogramDisplay({ spectrogram }) {
  if (!spectrogram) return null;
  
  return (
    <div className="spectrogram-display">
      <h2 className="spectrogram-display__title">
      Spectrogram
      </h2>
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
    </div>
  );
}