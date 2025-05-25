import { useState } from 'react';
import axios from 'axios';

export function usePredict() {
  const [predictions, setPredictions] = useState({});
  const [spectrogram, setSpectrogram] = useState('');
  const [numSegments, setNumSegments] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e, selectedFile, defaultFile) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      
      if (selectedFile) {
        console.log("Procesando archivo subido:", selectedFile.name);
        formData.append('file', selectedFile);
      } else if (defaultFile) {
        console.log("Procesando archivo predeterminado:", defaultFile);
        formData.append('default_file', defaultFile);
      } else {
        throw new Error('Por favor selecciona un archivo');
      }

      console.log("Enviando petición a /predict...");
      const res = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log("Respuesta recibida:", res.data);
      
      setPredictions(res.data.predictions);
      setSpectrogram(res.data.spectrogram);
      setNumSegments(res.data.num_segments);
      
    } catch (err) {
      console.error("Error en handleSubmit:", err);
      const errorMessage = err.response?.data?.detail || err.message || 'Error desconocido';
      setError(`Error procesando archivo: ${errorMessage}`);
      alert(`Error procesando archivo: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  return {
    predictions,
    spectrogram,
    numSegments,
    loading,
    error,
    handleSubmit,
  };
}