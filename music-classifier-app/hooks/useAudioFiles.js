import { useState, useEffect } from 'react';
import axios from 'axios';

export function useAudioFiles() {
  const [defaultFiles, setDefaultFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [defaultFile, setDefaultFile] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDefaultFiles = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await axios.get('http://localhost:8000/default-files');
        console.log("Respuesta completa: ", response);
        console.log("Datos de respuesta: ", response.data);
        
        if (response.data && Array.isArray(response.data.files)) {
          setDefaultFiles(response.data.files);
          console.log("Archivos cargados exitosamente:", response.data.files);
        } else {
          console.error("Formato inesperado:", response.data);
          setError("Formato de respuesta inesperado");
        }
      } catch (err) {
        console.error("Error al cargar archivos predeterminados:", err);
        setError(`Error al cargar archivos: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    fetchDefaultFiles();
  }, []);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    console.log("Archivo seleccionado:", file);
    setSelectedFile(file);
    setDefaultFile(''); // Limpiar selección de archivo predeterminado
  };

  const handleDefaultFileChange = (e) => {
    const fileName = e.target.value;
    console.log("Archivo predeterminado seleccionado:", fileName);
    setDefaultFile(fileName);
    setSelectedFile(null); // Limpiar selección de archivo subido
  };

  // Función para hacer la predicción - CORREGIDA
  const makePrediction = async () => {
    try {
      setLoading(true);
      setError(null);

      let response;

      if (selectedFile) {
        // Subir archivo usando FormData
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        console.log("Enviando archivo subido:", selectedFile.name);
        response = await axios.post('http://localhost:8000/predict', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
      } else if (defaultFile) {
        // Usar archivo predeterminado - USAR FormData también
        const formData = new FormData();
        formData.append('default_file', defaultFile);
        
        console.log("Enviando archivo predeterminado:", defaultFile);
        response = await axios.post('http://localhost:8000/predict', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
      } else {
        throw new Error("No se ha seleccionado ningún archivo");
      }

      console.log("Respuesta de predicción:", response.data);
      return response.data;

    } catch (err) {
      console.error("Error en predicción:", err);
      const errorMessage = err.response?.data?.detail || err.message || "Error desconocido";
      setError(`Error en predicción: ${errorMessage}`);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return {
    defaultFiles,
    selectedFile,
    defaultFile,
    loading,
    error,
    handleFileChange,
    handleDefaultFileChange,
    makePrediction,
  };
}