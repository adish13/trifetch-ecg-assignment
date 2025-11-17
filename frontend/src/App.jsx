import { useState, useEffect } from 'react'
import './App.css'
import EpisodeList from './components/EpisodeList'
import ECGViewer from './components/ECGViewer'
import SimpleECGViewer from './components/SimpleECGViewer'
import ErrorBoundary from './components/ErrorBoundary'
import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000'

function App() {
  const [episodes, setEpisodes] = useState([])
  const [selectedEpisode, setSelectedEpisode] = useState(null)
  const [ecgData, setEcgData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Fetch episodes on mount
  useEffect(() => {
    fetchEpisodes()
  }, [])

  const fetchEpisodes = async () => {
    try {
      setLoading(true)
      const response = await axios.get(`${API_BASE_URL}/episodes`)
      setEpisodes(response.data)
      setError(null)
    } catch (err) {
      setError('Failed to fetch episodes. Make sure the backend is running.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleEpisodeSelect = async (episode) => {
    setSelectedEpisode(episode)
    setLoading(true)
    setError(null)

    try {
      console.log('Fetching ECG data for episode:', episode.id)
      const response = await axios.get(`${API_BASE_URL}/episodes/${episode.id}/ecg`)
      console.log('ECG data received:', response.data)
      setEcgData(response.data)
    } catch (err) {
      setError(`Failed to load ECG data for episode ${episode.id}: ${err.message}`)
      console.error('Error loading ECG data:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>ECG Arrhythmia Detection</h1>
        <p className="subtitle">TriFetch Take-Home Assignment</p>
      </header>

      <div className="app-content">
        <aside className="sidebar">
          <h2>Episodes</h2>
          {error && <div className="error-message">{error}</div>}
          <EpisodeList
            episodes={episodes}
            selectedEpisode={selectedEpisode}
            onSelectEpisode={handleEpisodeSelect}
            loading={loading}
          />
        </aside>

        <main className="main-content">
          {loading && <div className="loading">Loading ECG data...</div>}
          {error && !loading && <div className="error-message">{error}</div>}
          {!loading && !ecgData && !selectedEpisode && (
            <div className="empty-state">
              <p>Select an episode from the list to view ECG data</p>
            </div>
          )}
          {!loading && ecgData && selectedEpisode && (
            <ErrorBoundary>
              <ECGViewer
                ecgData={ecgData}
                episode={selectedEpisode}
              />
            </ErrorBoundary>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
