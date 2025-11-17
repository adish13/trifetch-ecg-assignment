import React, { useState, useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, ReferenceArea } from 'recharts'
import './ECGViewer.css'

const ECGViewer = ({ ecgData, episode }) => {
  // State for the 8-second window position (in seconds)
  const [windowStart, setWindowStart] = useState(0)
  const windowDuration = 8 // 8 seconds as per requirements

  // Downsample factors
  const detailDownsample = 2 // For 8-second detailed view
  const overviewDownsample = 20 // For 90-second overview

  // Prepare full overview data (downsampled heavily for performance)
  const overviewData = useMemo(() => {
    if (!ecgData || !ecgData.channel1 || !ecgData.channel2) return []

    const data = []
    const ch1 = ecgData.channel1
    const ch2 = ecgData.channel2

    for (let i = 0; i < ch1.length; i += overviewDownsample) {
      data.push({
        index: i,
        time: i / ecgData.sampling_rate,
        ch1: ch1[i],
        ch2: ch2[i]
      })
    }

    return data
  }, [ecgData, overviewDownsample])

  // Prepare detailed view data (8-second window)
  const detailData = useMemo(() => {
    if (!ecgData || !ecgData.channel1 || !ecgData.channel2) return []

    const startIndex = Math.floor(windowStart * ecgData.sampling_rate)
    const endIndex = Math.floor((windowStart + windowDuration) * ecgData.sampling_rate)

    const data = []
    const ch1 = ecgData.channel1
    const ch2 = ecgData.channel2

    for (let i = startIndex; i < Math.min(endIndex, ch1.length); i += detailDownsample) {
      data.push({
        index: i,
        time: i / ecgData.sampling_rate,
        ch1: ch1[i],
        ch2: ch2[i]
      })
    }

    return data
  }, [ecgData, windowStart, windowDuration, detailDownsample])

  if (!ecgData) {
    return <div className="ecg-viewer">Loading ECG data...</div>
  }

  if (!overviewData || overviewData.length === 0) {
    return <div className="ecg-viewer">No chart data available</div>
  }

  const totalDuration = ecgData.channel1.length / ecgData.sampling_rate
  const eventTimeSeconds = ecgData && ecgData.event_index && ecgData.sampling_rate
    ? ecgData.event_index / ecgData.sampling_rate
    : 0
  const predictedEventTimeSeconds = ecgData?.predicted_event_index && ecgData?.sampling_rate
    ? ecgData.predicted_event_index / ecgData.sampling_rate
    : null

  // Handler for slider change
  const handleSliderChange = (e) => {
    const newStart = parseFloat(e.target.value)
    setWindowStart(newStart)
  }

  // Handler for clicking on overview chart to jump to that position
  const handleOverviewClick = (e) => {
    if (e && e.activeLabel !== undefined) {
      const clickedTime = parseFloat(e.activeLabel)
      const newStart = Math.max(0, Math.min(clickedTime - windowDuration / 2, totalDuration - windowDuration))
      setWindowStart(newStart)
    }
  }

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

  const windowEnd = windowStart + windowDuration

  try {
    return (
      <div className="ecg-viewer">
        {/* Header with episode info */}
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
                <span className="label">Duration:</span>
                <span className="value">{totalDuration.toFixed(1)}s (viewing {windowStart.toFixed(1)}s - {windowEnd.toFixed(1)}s)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Detailed 8-second view */}
        <div className="detail-view">
          <h3>Detailed View (8 seconds)</h3>
        <h3>Channel 1 (Lead I)</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            syncId="ecgSync"
            onMouseMove={(e) => {
              if (e && e.activeTooltipIndex !== undefined) {
                setHoveredIndex(e.activeTooltipIndex * downsampleFactor)
              }
            }}
            onMouseLeave={() => setHoveredIndex(null)}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis
              dataKey="time"
              label={{ value: 'Time (seconds)', position: 'insideBottom', offset: -5 }}
              tickFormatter={formatTime}
            />
            <YAxis
              label={{ value: 'Amplitude', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Event marker */}
            <ReferenceLine
              x={eventTimeSeconds}
              stroke="#ff4444"
              strokeWidth={2}
              label={{ value: 'Event', position: 'top', fill: '#ff4444' }}
            />

            {/* Predicted event marker */}
            {predictedEventTimeSeconds && (
              <ReferenceLine
                x={predictedEventTimeSeconds}
                stroke="#4444ff"
                strokeWidth={2}
                strokeDasharray="5 5"
                label={{ value: 'Predicted', position: 'top', fill: '#4444ff' }}
              />
            )}

            {/* Hover highlight region */}
            {hoveredIndex !== null && ecgData?.sampling_rate && (
              <ReferenceLine
                x={hoveredIndex / ecgData.sampling_rate}
                stroke="#2196F3"
                strokeWidth={1}
                strokeOpacity={0.5}
              />
            )}

            <Line
              type="monotone"
              dataKey="ch1"
              stroke="#2196F3"
              dot={false}
              strokeWidth={1.5}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>

        <h3>Channel 2 (Lead II)</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis
              dataKey="time"
              label={{ value: 'Time (seconds)', position: 'insideBottom', offset: -5 }}
              tickFormatter={formatTime}
            />
            <YAxis
              label={{ value: 'Amplitude', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip content={<CustomTooltip />} />

            <ReferenceLine
              x={eventTimeSeconds}
              stroke="#ff4444"
              strokeWidth={2}
            />

            {predictedEventTimeSeconds && (
              <ReferenceLine
                x={predictedEventTimeSeconds}
                stroke="#4444ff"
                strokeWidth={2}
                strokeDasharray="5 5"
              />
            )}

            <Line
              type="monotone"
              dataKey="ch2"
              stroke="#4CAF50"
              dot={false}
              strokeWidth={1.5}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
    )
  } catch (error) {
    console.error('Error rendering ECG viewer:', error)
    return (
      <div className="ecg-viewer" style={{ padding: '2rem' }}>
        <h2>Error displaying ECG data</h2>
        <p style={{ color: 'red' }}>{error.message}</p>
        <pre style={{ fontSize: '0.8rem', background: '#f5f5f5', padding: '1rem', overflow: 'auto' }}>
          {error.stack}
        </pre>
      </div>
    )
  }
}

export default ECGViewer

