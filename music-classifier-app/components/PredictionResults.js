import { Pie } from 'react-chartjs-2';

export default function PredictionResults({ predictions }) {
  if (Object.keys(predictions).length === 0) return null;

  const pieCharts = Object.entries(predictions).map(([model, probs]) => {
    const data = {
      labels: Object.keys(probs),
      datasets: [{
        data: Object.values(probs),
        backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF'],
      }],
    };
    return { model, data };
  });

  return (
    <div>
      <h2 className="text-xl mb-2">Prediction Results (Averaged Across Segments)</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {pieCharts.map(({ model, data }) => (
          <div key={model} className="border p-4">
            <h3 className="text-lg font-semibold">{model}</h3>
            <Pie data={data} />
          </div>
        ))}
      </div>
    </div>
  );
}