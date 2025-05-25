export default function SpectrogramDisplay({ spectrogram }) {
  if (!spectrogram) return null;
  return (
    <div className="mb-4">
      <h2 className="text-xl mb-2">Spectrogram</h2>
      <img src={spectrogram} alt="Spectrogram" className="max-w-full h-auto" />
    </div>
  );
}