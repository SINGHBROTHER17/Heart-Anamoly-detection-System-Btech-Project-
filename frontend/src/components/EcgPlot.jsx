import Plot from 'react-plotly.js';

/**
 * EcgPlot — renders either a single lead (mini-preview) or the full 12-lead
 * clinical layout (3 rows × 4 columns) depending on `layout="single"|"full"`.
 *
 * Input `signal` is a per-lead object: { I: [samples], II: [samples], ... }
 */
export default function EcgPlot({
  signal,
  fs = 500,
  layout = 'single',
  title = '',
  height,
  isDark = false,
}) {
  const leadNames = Object.keys(signal);
  const N = signal[leadNames[0]]?.length || 0;
  const time = Array.from({ length: N }, (_, i) => i / fs);

  const baseLayout = {
    autosize: true,
    margin: { l: 36, r: 8, t: 24, b: 28 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: isDark ? '#cbd5e1' : '#334155', size: 10 },
    xaxis: {
      gridcolor: isDark ? '#334155' : '#e2e8f0',
      zerolinecolor: isDark ? '#475569' : '#cbd5e1',
    },
    yaxis: {
      gridcolor: isDark ? '#334155' : '#e2e8f0',
      zerolinecolor: isDark ? '#475569' : '#cbd5e1',
    },
    showlegend: false,
  };

  const plotConfig = { responsive: true, displayModeBar: 'hover', displaylogo: false };

  if (layout === 'single') {
    return (
      <Plot
        data={[
          {
            x: time,
            y: signal[leadNames[0]],
            type: 'scatter',
            mode: 'lines',
            line: { color: '#2563eb', width: 1 },
          },
        ]}
        layout={{
          ...baseLayout,
          title: title || leadNames[0],
          height: height || 120,
          xaxis: { ...baseLayout.xaxis, title: '' },
          yaxis: { ...baseLayout.yaxis, title: '' },
        }}
        config={plotConfig}
        style={{ width: '100%' }}
        useResizeHandler
      />
    );
  }

  // Full 12-lead — 6 rows × 2 columns stays usable on mobile.
  // Standard clinical layout (3×4) only fits ≥ 900 px screens, so we adapt.
  const nRows = 6;
  const nCols = 2;
  const traces = [];
  const annotations = [];
  for (let i = 0; i < leadNames.length; i++) {
    const name = leadNames[i];
    const row = Math.floor(i / nCols) + 1;
    const col = (i % nCols) + 1;
    traces.push({
      x: time,
      y: signal[name],
      type: 'scatter',
      mode: 'lines',
      line: { color: '#2563eb', width: 0.8 },
      xaxis: `x${i === 0 ? '' : i + 1}`,
      yaxis: `y${i === 0 ? '' : i + 1}`,
      name,
    });
    annotations.push({
      text: name,
      font: { size: 11, weight: 600, color: isDark ? '#e2e8f0' : '#0f172a' },
      showarrow: false,
      xref: `x${i === 0 ? '' : i + 1} domain`,
      yref: `y${i === 0 ? '' : i + 1} domain`,
      x: 0.02, y: 0.92, xanchor: 'left', yanchor: 'top',
    });
  }

  // Shared-axis grid layout.
  const grid = {
    rows: nRows,
    columns: nCols,
    pattern: 'independent',
    roworder: 'top to bottom',
  };

  return (
    <Plot
      data={traces}
      layout={{
        ...baseLayout,
        grid,
        title: title || '',
        height: height || 700,
        annotations,
        margin: { l: 32, r: 8, t: 24, b: 24 },
      }}
      config={plotConfig}
      style={{ width: '100%' }}
      useResizeHandler
    />
  );
}
