import React from 'react'

const SimpleECGViewer = ({ ecgData, episode }) => {
  if (!ecgData) {
    return <div>No ECG data</div>
  }

  return (
    <div style={{ padding: '2rem', background: 'white', borderRadius: '8px' }}>
      <h2>Episode {episode.id}</h2>
      <div>
        <p><strong>Event:</strong> {episode.event_name}</p>
        <p><strong>Status:</strong> {episode.is_rejected ? 'Rejected' : 'Approved'}</p>
        <p><strong>Channel 1 samples:</strong> {ecgData.channel1?.length || 0}</p>
        <p><strong>Channel 2 samples:</strong> {ecgData.channel2?.length || 0}</p>
        <p><strong>Sampling rate:</strong> {ecgData.sampling_rate} Hz</p>
        <p><strong>Event index:</strong> {ecgData.event_index}</p>
        <p><strong>Predicted event index:</strong> {ecgData.predicted_event_index || 'N/A'}</p>
      </div>
      <div style={{ marginTop: '2rem' }}>
        <h3>First 10 samples (Channel 1):</h3>
        <pre>{JSON.stringify(ecgData.channel1?.slice(0, 10), null, 2)}</pre>
      </div>
    </div>
  )
}

export default SimpleECGViewer

