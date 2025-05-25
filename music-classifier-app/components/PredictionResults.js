import { Pie } from 'react-chartjs-2';
import '../styles/PredictionResults.css';

export default function PredictionResults({ predictions }) {
  if (Object.keys(predictions).length === 0) return null;

  const coolColors = [
    '#0ea5e9', 
    '#3b82f6', 
    '#8b5cf6', 
    '#06b6d4', 
    '#10b981',
    '#6366f1', 
    '#14b8a6', 
    '#84cc16', 
  ];

  const pieCharts = Object.entries(predictions).map(([model, probs]) => {
    const probEntries = Object.entries(probs);
    const maxProb = Math.max(...Object.values(probs));
    const maxLabel = probEntries.find(([, value]) => value === maxProb)?.[0];
    
    const data = {
      labels: Object.keys(probs),
      datasets: [{
        data: Object.values(probs),
        backgroundColor: coolColors.slice(0, Object.keys(probs).length),
        borderColor: '#ffffff',
        borderWidth: 3,
        hoverBorderWidth: 4,
        hoverBorderColor: '#1e293b',
      }],
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            padding: 20,
            usePointStyle: true,
            pointStyle: 'circle',
            font: {
              size: 12,
              weight: '500',
            },
            color: '#475569',
          },
        },
        tooltip: {
          backgroundColor: 'rgba(30, 41, 59, 0.9)',
          titleColor: '#ffffff',
          bodyColor: '#cbd5e1',
          borderColor: '#64748b',
          borderWidth: 1,
          cornerRadius: 8,
          displayColors: true,
          callbacks: {
            label: function(context) {
              const percentage = (context.parsed * 100).toFixed(1);
              return `${context.label}: ${percentage}%`;
            }
          }
        },
      },
      animation: {
        animateRotate: true,
        animateScale: true,
        duration: 1500,
        easing: 'easeInOutCubic',
      },
      layout: {
        padding: 10,
      },
    };

    return { 
      model, 
      data, 
      options, 
      maxProb: (maxProb * 100).toFixed(1), 
      maxLabel,
      stats: probEntries.map(([label, value]) => ({
        label,
        value: (value * 100).toFixed(1)
      })).sort((a, b) => parseFloat(b.value) - parseFloat(a.value))
    };
  });

  return (
    <div className="prediction-results">
      <h2 className="prediction-results__title">
       Prediction Results
        <br />
        <span style={{ fontSize: '0.875rem', fontWeight: '400', color: '#64748b' }}>
          (Averaged Across Segments)
        </span>
      </h2>
      
      <div className="prediction-results__grid">
        {pieCharts.map(({ model, data, options, maxProb, maxLabel, stats }) => (
          <div key={model} className="prediction-results__card">
            <div className="prediction-results__confidence">
              {maxProb}% confident
            </div>
            
            <h3 className="prediction-results__card-title">
              {model}
            </h3>
            
            <div className="prediction-results__chart-container">
              <div className="prediction-results__chart-wrapper">
                <Pie data={data} options={options} />
              </div>
            </div>
            
            <div className="prediction-results__stats">
              <div className="prediction-results__stat-item">
                <span className="prediction-results__stat-label">
                  Top Prediction:
                </span>
                <span className="prediction-results__stat-value">
                  {maxLabel}
                </span>
              </div>
              
              {stats.slice(0, 3).map(({ label, value }) => (
                <div key={label} className="prediction-results__stat-item">
                  <span className="prediction-results__stat-label">
                    {label}:
                  </span>
                  <span className="prediction-results__stat-value">
                    {value}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}