import '../styles/PredictionResults.css';

export default function SegmentInfo({ numSegments, audioUrl }) {
  if (numSegments === 0) return null;
  
  return (
    <div className="segment-info">
      <p className="segment-info__text">
        Processed 
        <span className="segment-info__count">
          {numSegments}
        </span>
        segment{numSegments > 1 ? 's' : ''} of 3 seconds each.
      </p>
      {audioUrl && (
        <audio controls style={{ marginTop: '1rem' }}>
          <source src={audioUrl} />
          Your browser does not support the audio element.
        </audio>
      )}
    </div>
  );
}