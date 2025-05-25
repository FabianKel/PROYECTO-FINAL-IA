export default function SegmentInfo({ numSegments }) {
  if (numSegments === 0) return null;
  return (
    <div className="mb-4">
      <p>Processed {numSegments} segment{numSegments > 1 ? 's' : ''} of 3 seconds each.</p>
    </div>
  );
}