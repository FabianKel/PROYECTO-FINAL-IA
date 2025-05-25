import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import FileSelector from '../components/FileSelector';
import SpectrogramDisplay from '../components/SpectrogramDisplay';
import PredictionResults from '../components/PredictionResults';
import SegmentInfo from '../components/SegmentInfo';
import { useAudioFiles } from '../hooks/useAudioFiles';
import { usePredict } from '../hooks/usePredict';

ChartJS.register(ArcElement, Tooltip, Legend);

export default function Home() {
  const { defaultFiles, selectedFile, defaultFile, handleFileChange, handleDefaultFileChange } = useAudioFiles();
  const { predictions, spectrogram, numSegments, loading, handleSubmit } = usePredict();

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-4">Song Genre Classifier</h1>
      <FileSelector
        defaultFiles={defaultFiles}
        selectedFile={selectedFile}
        defaultFile={defaultFile}
        onFileChange={handleFileChange}
        onDefaultFileChange={handleDefaultFileChange}
        onSubmit={(e) => handleSubmit(e, selectedFile, defaultFile)}
        loading={loading}
      />
      <SegmentInfo numSegments={numSegments} />
      <SpectrogramDisplay spectrogram={spectrogram} />
      <PredictionResults predictions={predictions} />
    </div>
  );
}