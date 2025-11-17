import React from 'react'
import './EpisodeList.css'

const EpisodeList = ({ episodes, selectedEpisode, onSelectEpisode, loading }) => {
  const getEventBadgeClass = (eventName) => {
    const eventType = eventName.toUpperCase()
    if (eventType.includes('AF')) return 'badge-af'
    if (eventType.includes('VTACH')) return 'badge-vtach'
    if (eventType.includes('PAUSE')) return 'badge-pause'
    return 'badge-default'
  }

  const formatDateTime = (dateTimeStr) => {
    const date = new Date(dateTimeStr)
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  if (loading && episodes.length === 0) {
    return <div className="loading-episodes">Loading episodes...</div>
  }

  return (
    <div className="episode-list">
      {episodes.map((episode) => (
        <div
          key={episode.id}
          className={`episode-item ${selectedEpisode?.id === episode.id ? 'selected' : ''}`}
          onClick={() => onSelectEpisode(episode)}
        >
          <div className="episode-header">
            <span className={`event-badge ${getEventBadgeClass(episode.event_name)}`}>
              {episode.event_name}
            </span>
            {episode.is_rejected && (
              <span className="rejected-badge">Rejected</span>
            )}
          </div>
          <div className="episode-id">ID: {episode.id}</div>
          <div className="episode-time">{formatDateTime(episode.event_time)}</div>
          <div className="episode-patient">Patient: {episode.patient_id.substring(0, 8)}...</div>
        </div>
      ))}
    </div>
  )
}

export default EpisodeList

