import { useState } from 'react';

import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import FileSelector from '../components/FileSelector';
import SpectrogramDisplay from '../components/SpectrogramDisplay';
import PredictionResults from '../components/PredictionResults';
import GenreSegmentChart from "@/components/GenreSegmentChart";
import SegmentInfo from '../components/SegmentInfo';
import { useAudioFiles } from '../hooks/useAudioFiles';
import { usePredict } from '../hooks/usePredict';

ChartJS.register(ArcElement, Tooltip, Legend);

export default function Home() {
  const { defaultFiles, selectedFile, defaultFile, handleFileChange, handleDefaultFileChange } = useAudioFiles();
  const { predictions, segmentPredictions, spectrogram, numSegments, loading, handleSubmit } = usePredict();


  return (
    <div className="container">
<img 
  src="song-genre-classifier.png" 
  alt="Song Genre Classifier" 
  style={{ display: 'block', margin: '0 auto', width: '1200px', height: '160px' }}
/>
      
        <FileSelector
        defaultFiles={defaultFiles}
        selectedFile={selectedFile}
        defaultFile={defaultFile}
        onFileChange={handleFileChange}
        onDefaultFileChange={handleDefaultFileChange}
        onSubmit={(e, selectedFile, defaultFile, predictOptions) =>  handleSubmit(e, selectedFile, defaultFile, predictOptions)}
        loading={loading}
      />

      <SegmentInfo numSegments={numSegments} />
      <SpectrogramDisplay spectrogram={spectrogram} />

      {/*<GenreSegmentChart segmentPredictions={segmentPredictions}/>*/}
      
      <PredictionResults predictions={predictions} />
    </div>
  );
}