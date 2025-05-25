import { useState, useEffect } from 'react';
import axios from 'axios';

export function useAudioFiles() {
  const [defaultFiles, setDefaultFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [defaultFile, setDefaultFile] = useState('');

  useEffect(() => {
    axios.get('http://localhost:8000/default-files')
      .then(res => {
        console.log("Respuesta: ", res.data);
        if (res.data && Array.isArray(res.data.files)) {
          setDefaultFiles(res.data.files);
        } else {
          console.error("Formato inesperado:", res.data);
        }
      })
      .catch(err => console.error(err));
  }, []);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setDefaultFile('');
  };

  const handleDefaultFileChange = (e) => {
    setDefaultFile(e.target.value);
    setSelectedFile(null);
  };

  return {
    defaultFiles,
    selectedFile,
    defaultFile,
    handleFileChange,
    handleDefaultFileChange,
  };
}