import { useState } from 'react';
import axios from 'axios';

export function usePredict() {
  const [predictions, setPredictions] = useState({});
  const [spectrogram, setSpectrogram] = useState('');
  const [numSegments, setNumSegments] = useState(0);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e, selectedFile, defaultFile) => {
    e.preventDefault();
    setLoading(true);
    const formData = new FormData();
    if (selectedFile) {
      formData.append('file', selectedFile);
    } else if (defaultFile) {
      formData.append('default_file', defaultFile);
    } else {
      alert('Please select a file');
      setLoading(false);
      return;
    }

    try {
      const res = await axios.post('http://localhost:8000/predict', formData);
      setPredictions(res.data.predictions);
      setSpectrogram(res.data.spectrogram);
      setNumSegments(res.data.num_segments);
    } catch (err) {
      console.error(err);
      alert('Error processing file');
    }
    setLoading(false);
  };

  return {
    predictions,
    spectrogram,
    numSegments,
    loading,
    handleSubmit,
  };
}