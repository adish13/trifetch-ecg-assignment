import React, { useState, useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, ReferenceArea } from 'recharts'
import './ECGViewer.css'

const ECGViewer = ({ ecgData, episode }) => {
  // Calculate event time first to center the initial view
  const eventTimeSeconds = ecgData?.event_index && ecgData?.sampling_rate
    ? ecgData.event_index / ecgData.sampling_rate
    : 0

  const windowDuration = 8 // 8 seconds as per requirements

  // Center the initial window around the event (4 seconds before event)
  const initialWindowStart = Math.max(0, eventTimeSeconds - 4)

  // State for the 8-second window position (in seconds)
  const [windowStart, setWindowStart] = useState(initialWindowStart)

  if (!ecgData || !ecgData.channel1 || !ecgData.channel2) {
    return <div className="ecg-viewer">Loading ECG data...</div>
  }

  const totalDuration = ecgData.channel1.length / ecgData.sampling_rate
  const windowEnd = windowStart + windowDuration

  // Predicted event time
  const predictedEventTimeSeconds = ecgData.predicted_event_index
    ? ecgData.predicted_event_index / ecgData.sampling_rate
    : null

  // Prepare full overview data (heavily downsampled)
  const overviewData = useMemo(() => {
    const data = []
    const downsample = 20
    for (let i = 0; i < ecgData.channel1.length; i += downsample) {
      data.push({
        time: i / ecgData.sampling_rate,
        ch1: ecgData.channel1[i],
        ch2: ecgData.channel2[i]
      })
    }
    return data
  }, [ecgData])

  // Prepare detailed view data (8-second window)
  const detailData = useMemo(() => {
    const startIndex = Math.floor(windowStart * ecgData.sampling_rate)
    const endIndex = Math.floor((windowStart + windowDuration) * ecgData.sampling_rate)
    const data = []
    const downsample = 2

    for (let i = startIndex; i < Math.min(endIndex, ecgData.channel1.length); i += downsample) {
      data.push({
        time: i / ecgData.sampling_rate,
        ch1: ecgData.channel1[i],
        ch2: ecgData.channel2[i]
      })
    }
    return data
  }, [ecgData, windowStart, windowDuration])

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = (seconds % 60).toFixed(1)
    return `${mins}:${secs.padStart(4, '0')}`
  }

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length > 0) {
      const data = payload[0].payload
      return (
        <div className="custom-tooltip">
          <p className="time">Time: {formatTime(data.time)}</p>
          <p className="ch1">Ch1: {data.ch1?.toFixed(0) || 'N/A'}</p>
          <p className="ch2">Ch2: {data.ch2?.toFixed(0) || 'N/A'}</p>
        </div>
      )
    }
    return null
  }

  const handleSliderChange = (e) => {
    setWindowStart(parseFloat(e.target.value))
  }

  const handleOverviewClick = (e) => {
    if (e && e.activeLabel !== undefined) {
      const clickedTime = parseFloat(e.activeLabel)
      const newStart = Math.max(0, Math.min(clickedTime - windowDuration / 2, totalDuration - windowDuration))
      setWindowStart(newStart)
    }
  }

  return (
    <div className="ecg-viewer">
      {/* Header */}
      <div className="ecg-header">
        <div className="episode-info">
          <h2>Episode {episode?.id || 'Unknown'}</h2>
          <div className="info-grid">
            <div className="info-item">
              <span className="label">Event Type:</span>
              <span className={`value event-${episode?.event_name?.toLowerCase() || 'unknown'}`}>
                {episode?.event_name || 'Unknown'}
              </span>
            </div>
            <div className="info-item">
              <span className="label">Status:</span>
              <span className={`value ${episode?.is_rejected ? 'rejected' : 'approved'}`}>
                {episode?.is_rejected ? 'Rejected' : 'Approved'}
              </span>
            </div>
            <div className="info-item">
              <span className="label">Event Time:</span>
              <span className="value">{formatTime(eventTimeSeconds)}</span>
            </div>
            <div className="info-item">
              <span className="label">Viewing:</span>
              <span className="value">{formatTime(windowStart)} - {formatTime(windowEnd)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Detailed 8-second view - Channel 1 */}
      <div className="detail-view">
        <h3>Detailed View (8 seconds) - Channel 1</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={detailData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis dataKey="time" domain={[windowStart, windowEnd]} type="number" tickFormatter={formatTime} />
            <YAxis label={{ value: 'Ch1', angle: -90, position: 'insideLeft' }} />
            <Tooltip content={<CustomTooltip />} />

            {eventTimeSeconds >= windowStart && eventTimeSeconds <= windowEnd && (
              <ReferenceLine x={eventTimeSeconds} stroke="#ff4444" strokeWidth={2}
                label={{ value: 'Event', position: 'top', fill: '#ff4444' }} />
            )}

            {predictedEventTimeSeconds && predictedEventTimeSeconds >= windowStart && predictedEventTimeSeconds <= windowEnd && (
              <ReferenceLine x={predictedEventTimeSeconds} stroke="#4444ff" strokeWidth={2} strokeDasharray="5 5"
                label={{ value: 'Predicted', position: 'top', fill: '#4444ff' }} />
            )}

            <Line type="monotone" dataKey="ch1" stroke="#2196F3" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>

        {/* Channel 2 */}
        <h3 style={{ marginTop: '1rem' }}>Detailed View (8 seconds) - Channel 2</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={detailData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis dataKey="time" domain={[windowStart, windowEnd]} type="number" tickFormatter={formatTime} />
            <YAxis label={{ value: 'Ch2', angle: -90, position: 'insideLeft' }} />
            <Tooltip content={<CustomTooltip />} />

            {eventTimeSeconds >= windowStart && eventTimeSeconds <= windowEnd && (
              <ReferenceLine x={eventTimeSeconds} stroke="#ff4444" strokeWidth={2} />
            )}

            {predictedEventTimeSeconds && predictedEventTimeSeconds >= windowStart && predictedEventTimeSeconds <= windowEnd && (
              <ReferenceLine x={predictedEventTimeSeconds} stroke="#4444ff" strokeWidth={2} strokeDasharray="5 5" />
            )}

            <Line type="monotone" dataKey="ch2" stroke="#FF6B6B" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Slider control */}
      <div className="slider-control">
        <label>
          <strong>Navigate Timeline:</strong> {formatTime(windowStart)} - {formatTime(windowEnd)} of {formatTime(totalDuration)}
        </label>
        <input
          type="range"
          min="0"
          max={Math.max(0, totalDuration - windowDuration)}
          step="0.1"
          value={windowStart}
          onChange={handleSliderChange}
          className="time-slider"
        />
      </div>

      {/* Overview - Full 90 seconds */}
      <div className="overview-view">
        <h3>Full Recording Overview (90 seconds) - Click to jump to position</h3>
        <ResponsiveContainer width="100%" height={120}>
          <LineChart data={overviewData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }} onClick={handleOverviewClick}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis dataKey="time" tickFormatter={formatTime} />
            <YAxis hide />

            {/* Highlight the current 8-second window */}
            <ReferenceArea
              x1={windowStart}
              x2={windowEnd}
              fill="#2196F3"
              fillOpacity={0.2}
              stroke="#2196F3"
              strokeWidth={2}
              label={{ value: 'Current View', position: 'top' }}
            />

            {/* Event marker */}
            <ReferenceLine
              x={eventTimeSeconds}
              stroke="#ff4444"
              strokeWidth={2}
              label={{ value: 'Event', position: 'top', fill: '#ff4444', fontSize: 10 }}
            />

            {/* Predicted event marker */}
            {predictedEventTimeSeconds && (
              <ReferenceLine
                x={predictedEventTimeSeconds}
                stroke="#4444ff"
                strokeWidth={2}
                strokeDasharray="5 5"
                label={{ value: 'Pred', position: 'bottom', fill: '#4444ff', fontSize: 10 }}
              />
            )}

            <Line type="monotone" dataKey="ch1" stroke="#666" dot={false} strokeWidth={1} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default ECGViewer


