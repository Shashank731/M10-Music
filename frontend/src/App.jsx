import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronUp,
  Disc3,
  LockKeyhole,
  LogOut,
  Music2,
  Pause,
  Play,
  Search,
  SkipBack,
  SkipForward,
  Sparkles,
  UserRound,
  X,
} from "lucide-react";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  const token = localStorage.getItem("token");
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(`${API_BASE_URL}${path}`, { ...opts, headers });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "Request failed");
  }
  return response.json();
}

function getAudioUrl(song) {
  if (!song) return "";
  if (song.spotify_preview_url) return song.spotify_preview_url;
  if (song.audio_source === "local_mp3") {
    const token = localStorage.getItem("token");
    return token
      ? `${API_BASE_URL}/songs/${encodeURIComponent(song.track_id)}/audio?token=${encodeURIComponent(token)}`
      : "";
  }
  return "";
}

function SongList({ songs, onSelect, activeId }) {
  if (!songs.length) {
    return (
      <div className="empty-state">
        <Music2 size={18} />
        <span>No tracks here yet.</span>
      </div>
    );
  }

  return (
    <div className="song-list">
      {songs.map((song) => (
        <button
          className={`song-row ${activeId === song.track_id ? "active" : ""}`}
          key={song.track_id}
          onClick={() => onSelect(song.track_id)}
        >
          <span className="track-icon">
            <Disc3 size={18} />
          </span>
          <span className="track-copy">
            <strong>{song.name || song.track_id}</strong>
            <small>{song.artist || "Unknown artist"}</small>
          </span>
          <Play size={16} className="row-play" />
        </button>
      ))}
    </div>
  );
}

function PlayerControls({ onPrevious, onTogglePlay, onNext, isPlaying, playDisabled }) {
  return (
    <div className="player-controls">
      <button onClick={onPrevious} className="round-button" aria-label="Previous track">
        <SkipBack size={18} />
      </button>
      <button
        onClick={onTogglePlay}
        className="round-button play-button"
        aria-label={isPlaying ? "Pause" : "Play"}
        disabled={playDisabled}
      >
        {isPlaying ? <Pause size={20} /> : <Play size={20} />}
      </button>
      <button onClick={onNext} className="round-button" aria-label="Next track">
        <SkipForward size={18} />
      </button>
    </div>
  );
}

function MiniPlayer({
  song,
  audioUrl,
  isPlaying,
  onTogglePlay,
  onPrevious,
  onNext,
  onExpand,
}) {
  if (!song) return null;

  return (
    <footer className="mini-player">
      <button className="mini-player-main" onClick={onExpand}>
        <span className="mini-art">
          <Disc3 size={22} />
        </span>
        <span className="mini-copy">
          <strong>{song.name || song.track_id}</strong>
          <small>{song.artist || "Unknown artist"}</small>
        </span>
        <ChevronUp size={18} className="mini-expand" />
      </button>
      <PlayerControls
        onPrevious={onPrevious}
        onTogglePlay={onTogglePlay}
        onNext={onNext}
        isPlaying={isPlaying}
        playDisabled={!audioUrl}
      />
    </footer>
  );
}

function ExpandedPlayer({
  song,
  audioUrl,
  isPlaying,
  onTogglePlay,
  onPrevious,
  onNext,
  onClose,
}) {
  if (!song) return null;

  return (
    <div className="modal-backdrop player-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className="expanded-player"
        role="dialog"
        aria-modal="true"
        aria-label="Song player"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <button className="icon-button close-button" onClick={onClose} aria-label="Close player">
          <X size={18} />
        </button>
        <div className="expanded-art">
          <Disc3 size={76} />
        </div>
        <div className="expanded-copy">
          <span>{song.artist || "Unknown artist"}</span>
          <h2>{song.name || song.track_id}</h2>
          <p>{song.year || "Year unavailable"}</p>
          <p>{song.tags || song.genre || "No tags available"}</p>
        </div>
        {!audioUrl && (
          <div className="locked-note">
            <LockKeyhole size={16} />
            <span>No playable audio source was found for this track.</span>
          </div>
        )}
        <PlayerControls
          onPrevious={onPrevious}
          onTogglePlay={onTogglePlay}
          onNext={onNext}
          isPlaying={isPlaying}
          playDisabled={!audioUrl}
        />
      </section>
    </div>
  );
}

function AuthModal({
  mode,
  setMode,
  email,
  setEmail,
  password,
  setPassword,
  onSubmit,
  onClose,
  message,
}) {
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className="auth-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Sign in"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <button className="icon-button close-button" onClick={onClose} aria-label="Close">
          <X size={18} />
        </button>
        <div className="modal-heading">
          <span className="modal-mark">
            <LockKeyhole size={18} />
          </span>
          <div>
            <h2>{mode === "login" ? "Welcome back" : "Create your account"}</h2>
            <p>Sign in to unlock search, recommendations, and playback.</p>
          </div>
        </div>

        <form onSubmit={onSubmit} className="auth-form">
          <div className="segmented-control">
            <button
              type="button"
              className={mode === "login" ? "active" : ""}
              onClick={() => setMode("login")}
            >
              Login
            </button>
            <button
              type="button"
              className={mode === "register" ? "active" : ""}
              onClick={() => setMode("register")}
            >
              Register
            </button>
          </div>
          <label>
            <span>Email</span>
            <input
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
              type="email"
              required
            />
          </label>
          <label>
            <span>Password</span>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Minimum 6 characters"
              required
            />
          </label>
          {message && <p className="form-note">{message}</p>}
          <button type="submit" className="primary full">
            {mode === "login" ? "Sign in" : "Create account"}
          </button>
        </form>
      </section>
    </div>
  );
}

function LandingPage({ onAuth }) {
  return (
    <main className="landing-shell">
      <nav className="landing-nav">
        <span className="brand">
          <Disc3 size={20} />
          M10 Music
        </span>
        <button className="ghost-button" onClick={onAuth}>
          Sign in
        </button>
      </nav>

      <section className="landing-hero">
        <div className="hero-copy">
          <span className="eyebrow">
            <Sparkles size={16} />
            Private music intelligence
          </span>
          <h1>Premium recommendations for listeners who want less noise.</h1>
          <p>
            Search the catalog, inspect song signals, and play matched tracks after
            signing in.
          </p>
          <div className="hero-actions">
            <button className="primary large" onClick={onAuth}>
              Sign in / Register
            </button>
          </div>
        </div>
        <div className="hero-player" aria-hidden="true">
          <div className="hero-disc">
            <Disc3 size={92} />
          </div>
          <div className="hero-track">
            <span>Recommendation engine</span>
            <strong>Ready when you are</strong>
          </div>
          <div className="hero-bars">
            <span />
            <span />
            <span />
            <span />
            <span />
          </div>
        </div>
      </section>
    </main>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [selectedSong, setSelectedSong] = useState(null);
  const [songs, setSongs] = useState([]);
  const [listTitle, setListTitle] = useState("Browse");
  const [listDescription, setListDescription] = useState("Search or pick a track to start a recommendation queue.");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [showPlayer, setShowPlayer] = useState(false);

  const [showAuth, setShowAuth] = useState(false);
  const [authMode, setAuthMode] = useState("login");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [userEmail, setUserEmail] = useState(
    localStorage.getItem("user_email") || "",
  );
  const audioRef = useRef(null);
  const shouldResumeRef = useRef(false);

  const audioUrl = useMemo(() => getAudioUrl(selectedSong), [selectedSong]);

  useEffect(() => {
    if (!userEmail) return;
    api("/search?limit=8")
      .then((data) => {
        setSongs(data.results);
        setListTitle("Browse");
        setListDescription("Search or pick a track to start a recommendation queue.");
      })
      .catch((err) => setError(err.message));
  }, [userEmail]);

  useEffect(() => {
    const shouldResume = shouldResumeRef.current;
    shouldResumeRef.current = false;
    setIsPlaying(false);
    if (shouldResume && audioRef.current && audioUrl) {
      audioRef.current.play().catch((err) => setError(err.message));
    }
  }, [audioUrl]);

  async function runSearch(event) {
    event.preventDefault();
    setError("");
    try {
      const data = await api(`/search?q=${encodeURIComponent(query)}&limit=12`);
      setSongs(data.results);
      setListTitle(query ? `Results for "${query}"` : "Browse");
      setListDescription("Choose a track to play it and generate similar songs.");
    } catch (err) {
      setError(err.message);
    }
  }

  async function loadSong(trackId, { refreshQueue = true } = {}) {
    setError("");
    setIsPlaying(false);
    try {
      const song = await api(`/songs/${trackId}`);
      setSelectedSong(song);
      if (refreshQueue) {
        const recs = await api(`/recommend/song/${trackId}?top_k=10`);
        setSongs(recs.recommendations);
        setListTitle("Recommended next");
        setListDescription(`A queue built from ${song.name || song.track_id}.`);
      }
    } catch (err) {
      setError(err.message);
    }
  }

  function selectSong(trackId) {
    loadSong(trackId, { refreshQueue: true });
  }

  async function recommendForUser(event) {
    event?.preventDefault?.();
    setError("");
    try {
      const data = await api(`/recommend/user/${encodeURIComponent(userEmail)}?top_k=10`);
      setSongs(data.recommendations);
      setListTitle("For you");
      setListDescription("Personalized recommendations from your listening profile.");
      setSelectedSong(null);
      setIsPlaying(false);
    } catch (err) {
      setError(err.message);
    }
  }

  function currentIndex() {
    if (!selectedSong) return -1;
    return songs.findIndex((song) => song.track_id === selectedSong.track_id);
  }

  function moveBy(offset, { resume = isPlaying } = {}) {
    if (!songs.length) return;
    const index = currentIndex();
    const nextIndex =
      index === -1
        ? offset > 0
          ? 0
          : songs.length - 1
        : (index + offset + songs.length) % songs.length;
    shouldResumeRef.current = resume;
    loadSong(songs[nextIndex].track_id, { refreshQueue: false });
  }

  function togglePlay() {
    const audio = audioRef.current;
    if (!audio || !audioUrl) return;
    if (audio.paused) {
      audio.play().catch((err) => setError(err.message));
    } else {
      audio.pause();
    }
  }

  async function submitAuth(event) {
    event.preventDefault();
    setError("");
    setNotice("");
    try {
      if (authMode === "login") {
        const email = authEmail.trim().toLowerCase();
        const data = await api("/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password: authPassword }),
        });
        localStorage.setItem("token", data.access_token);
        localStorage.setItem("user_email", email);
        setUserEmail(email);
        setShowAuth(false);
        setAuthPassword("");
      } else {
        await api("/auth/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email: authEmail.trim().toLowerCase(),
            password: authPassword,
          }),
        });
        setAuthMode("login");
        setNotice("Account created. Sign in to continue.");
      }
    } catch (err) {
      setNotice(err.message);
    }
  }

  function logout() {
    localStorage.removeItem("token");
    localStorage.removeItem("user_email");
    setUserEmail("");
    setSelectedSong(null);
    setSongs([]);
    setIsPlaying(false);
    setShowPlayer(false);
  }

  if (!userEmail) {
    return (
      <>
        <LandingPage onAuth={() => setShowAuth(true)} />
        {showAuth && (
          <AuthModal
            mode={authMode}
            setMode={setAuthMode}
            email={authEmail}
            setEmail={setAuthEmail}
            password={authPassword}
            setPassword={setAuthPassword}
            onSubmit={submitAuth}
            onClose={() => setShowAuth(false)}
            message={notice}
          />
        )}
      </>
    );
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <span className="brand small">
              <Disc3 size={18} />
              M10 Music
            </span>
            <h1>Music Recommendations</h1>
            <p>Search, inspect, and play songs from your recommendation workspace.</p>
          </div>

          <div className="user-pill">
            <UserRound size={16} />
            <span>{userEmail}</span>
            <button className="icon-button" onClick={logout} aria-label="Log out">
              <LogOut size={16} />
            </button>
          </div>
        </header>

        {error && <div className="alert">{error}</div>}

          <section className="panel queue-panel">
            <div className="panel-heading">
              <div>
                <h2>{listTitle}</h2>
                <p>{listDescription}</p>
              </div>
              <button onClick={recommendForUser} className="secondary-button">
                Recommend for me
              </button>
            </div>
            <form className="inline-form" onSubmit={runSearch}>
              <Search size={18} />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Song, artist, or tag"
              />
              <button type="submit">Search</button>
            </form>
            <SongList
              songs={songs}
              onSelect={selectSong}
              activeId={selectedSong?.track_id}
            />
          </section>
      </section>
      <audio
        className="transport-audio"
        ref={audioRef}
        key={audioUrl}
        src={audioUrl}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => {
          setIsPlaying(false);
          moveBy(1, { resume: true });
        }}
      />
      <MiniPlayer
        song={selectedSong}
        audioUrl={audioUrl}
        isPlaying={isPlaying}
        onTogglePlay={togglePlay}
        onPrevious={() => moveBy(-1)}
        onNext={() => moveBy(1)}
        onExpand={() => setShowPlayer(true)}
      />
      {showPlayer && (
        <ExpandedPlayer
          song={selectedSong}
          audioUrl={audioUrl}
          isPlaying={isPlaying}
          onTogglePlay={togglePlay}
          onPrevious={() => moveBy(-1)}
          onNext={() => moveBy(1)}
          onClose={() => setShowPlayer(false)}
        />
      )}
    </main>
  );
}
