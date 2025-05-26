import React from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const genreColors = {
  Rock: "#ef4444",
  Pop: "#3b82f6",
  Jazz: "#eab308",
  Classical: "#10b981",
  Reggaeton: "#8b5cf6",
  HipHop: "#f97316",
  Blues: "#06b6d4",
  Country: "#84cc16",
  Disco: "#ec4899",
  Metal: "#64748b"
};

function GenreSegmentChart({ segmentPredictions }) {
  const models = Object.keys(segmentPredictions);
  const numSegments = segmentPredictions[models[0]]?.length || 0;

  const data = {
    labels: Array.from({ length: numSegments }, (_, i) => `Segmento ${i + 1}`),
    datasets: models.map((model, idx) => ({
      label: model,
      data: segmentPredictions[model].map(() => 1),
      backgroundColor: segmentPredictions[model].map(genre => genreColors[genre] || "gray"),
      stack: "stack" + idx
    }))
  };

  const options = {
    indexAxis: "x",
    responsive: true,
    plugins: {
      legend: { display: true },
      tooltip: {
        callbacks: {
          label: function (context) {
            const genre = segmentPredictions[context.dataset.label][context.dataIndex];
            return `${context.dataset.label}: ${genre}`;
          }
        }
      },
      title: {
        display: true,
        text: "Género por Segmento y Modelo"
      }
    },
    scales: {
      y: {
        ticks: { display: false }, // ocultar ejes Y porque todas las barras valen 1
        grid: { display: false }
      },
      x: {
        stacked: true
      }
    }
  };

  return <Bar data={data} options={options} />;
}

export default GenreSegmentChart;
