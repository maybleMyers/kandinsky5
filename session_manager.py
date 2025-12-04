"""
Session Manager for K1 Gradio Application

Provides session persistence and reconnect capabilities:
- SessionManager: Persists UI state to disk
- ProcessRegistry: Session-aware subprocess tracking
- GenerationStatus: Tracks generation progress

Usage:
    from session_manager import session_manager, process_registry
"""

import uuid
import json
import threading
import subprocess
import os
import time
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List


@dataclass
class GenerationStatus:
    """Tracks the current state of a generation process."""
    state: str = 'idle'  # 'idle', 'generating', 'completed', 'error', 'stopped'
    current_step: int = 0
    total_steps: int = 0
    batch_index: int = 0
    batch_total: int = 0
    eta_seconds: Optional[float] = None
    preview_path: Optional[str] = None
    output_files: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    subprocess_pid: Optional[int] = None
    started_at: Optional[float] = None
    output_filename: Optional[str] = None
    last_progress_text: str = ""


@dataclass
class SessionState:
    """Complete state for a user session."""
    session_id: str
    created_at: float
    last_active: float
    # UI parameters snapshot (all input values)
    ui_params: Dict[str, Any] = field(default_factory=dict)
    # Generation status
    generation_status: GenerationStatus = field(default_factory=GenerationStatus)
    # History of generated videos [(path, label), ...]
    video_history: List[tuple] = field(default_factory=list)
    # Active tab: 'gen_tab' or 'v2v'
    active_tab: str = 'gen_tab'


class SessionManager:
    """
    Manages user sessions with file-based persistence.

    Sessions are stored in sessions/{session_id}/state.json
    """

    def __init__(self, sessions_dir: str = "sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self._active_sessions: Dict[str, SessionState] = {}
        self._load_existing_sessions()

    def _session_path(self, session_id: str) -> Path:
        """Get the directory path for a session."""
        return self.sessions_dir / session_id

    def _load_existing_sessions(self):
        """Load any sessions that were persisted to disk on startup."""
        if not self.sessions_dir.exists():
            return

        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir():
                state_file = session_dir / "state.json"
                if state_file.exists():
                    try:
                        with open(state_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Reconstruct GenerationStatus
                        gen_status_data = data.get('generation_status', {})
                        gen_status = GenerationStatus(
                            state=gen_status_data.get('state', 'idle'),
                            current_step=gen_status_data.get('current_step', 0),
                            total_steps=gen_status_data.get('total_steps', 0),
                            batch_index=gen_status_data.get('batch_index', 0),
                            batch_total=gen_status_data.get('batch_total', 0),
                            eta_seconds=gen_status_data.get('eta_seconds'),
                            preview_path=gen_status_data.get('preview_path'),
                            output_files=gen_status_data.get('output_files', []),
                            error_message=gen_status_data.get('error_message'),
                            subprocess_pid=gen_status_data.get('subprocess_pid'),
                            started_at=gen_status_data.get('started_at'),
                            output_filename=gen_status_data.get('output_filename'),
                            last_progress_text=gen_status_data.get('last_progress_text', '')
                        )

                        # Mark any "generating" sessions as interrupted on restart
                        if gen_status.state == 'generating':
                            gen_status.state = 'interrupted'

                        # Reconstruct video history as list of tuples
                        video_history = []
                        for item in data.get('video_history', []):
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                video_history.append((item[0], item[1]))
                            elif isinstance(item, dict):
                                video_history.append((item.get('path', ''), item.get('label', '')))

                        session = SessionState(
                            session_id=data['session_id'],
                            created_at=data['created_at'],
                            last_active=data['last_active'],
                            ui_params=data.get('ui_params', {}),
                            generation_status=gen_status,
                            video_history=video_history,
                            active_tab=data.get('active_tab', 'gen_tab')
                        )
                        self._active_sessions[session.session_id] = session
                        print(f"[SessionManager] Loaded session: {session.session_id}")
                    except Exception as e:
                        print(f"[SessionManager] Failed to load session {session_dir.name}: {e}")

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())[:8]  # Short ID for convenience
        now = time.time()

        with self._lock:
            session = SessionState(
                session_id=session_id,
                created_at=now,
                last_active=now,
            )
            self._active_sessions[session_id] = session
            self._persist_session(session)

        print(f"[SessionManager] Created new session: {session_id}")
        return session_id

    def get_or_create_session(self, session_id: Optional[str]) -> str:
        """Get existing session or create new one if not found."""
        with self._lock:
            if session_id and session_id in self._active_sessions:
                session = self._active_sessions[session_id]
                session.last_active = time.time()
                print(f"[SessionManager] Resumed session: {session_id}")
                return session_id
            return self.create_session()

    def save_ui_state(self, session_id: str, params: Dict[str, Any], tab: str = None):
        """Save current UI parameters for a session."""
        with self._lock:
            if session_id in self._active_sessions:
                session = self._active_sessions[session_id]
                session.ui_params = params
                session.last_active = time.time()
                if tab:
                    session.active_tab = tab
                self._persist_session(session)

    def update_generation_status(self, session_id: str, **kwargs):
        """Update generation progress for a session."""
        with self._lock:
            if session_id in self._active_sessions:
                session = self._active_sessions[session_id]
                for key, value in kwargs.items():
                    if hasattr(session.generation_status, key):
                        setattr(session.generation_status, key, value)
                session.last_active = time.time()
                self._persist_session(session)

    def add_to_history(self, session_id: str, video_path: str, label: str):
        """Add completed video to session history."""
        with self._lock:
            if session_id in self._active_sessions:
                session = self._active_sessions[session_id]
                session.video_history.append((video_path, label))
                session.last_active = time.time()
                self._persist_session(session)

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session state (for reconnect recovery)."""
        with self._lock:
            return self._active_sessions.get(session_id)

    def get_video_history(self, session_id: str) -> List[tuple]:
        """Get video history for a session."""
        with self._lock:
            session = self._active_sessions.get(session_id)
            if session:
                return session.video_history.copy()
            return []

    def clear_video_history(self, session_id: str):
        """Clear video history for a session."""
        with self._lock:
            if session_id in self._active_sessions:
                session = self._active_sessions[session_id]
                session.video_history = []
                self._persist_session(session)

    def _persist_session(self, session: SessionState):
        """Write session state to disk."""
        session_dir = self._session_path(session.session_id)
        session_dir.mkdir(exist_ok=True)

        state_file = session_dir / "state.json"
        data = {
            'session_id': session.session_id,
            'created_at': session.created_at,
            'last_active': session.last_active,
            'ui_params': session.ui_params,
            'generation_status': asdict(session.generation_status),
            'video_history': list(session.video_history),  # List of tuples -> list of lists
            'active_tab': session.active_tab
        }

        try:
            # Write atomically using temp file
            temp_file = state_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_file.replace(state_file)
        except Exception as e:
            print(f"[SessionManager] Failed to persist session {session.session_id}: {e}")

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours."""
        cutoff = time.time() - (max_age_hours * 3600)
        with self._lock:
            to_remove = []
            for sid, session in self._active_sessions.items():
                if session.last_active < cutoff:
                    to_remove.append(sid)

            for sid in to_remove:
                del self._active_sessions[sid]
                session_dir = self._session_path(sid)
                if session_dir.exists():
                    try:
                        shutil.rmtree(session_dir)
                        print(f"[SessionManager] Cleaned up old session: {sid}")
                    except Exception as e:
                        print(f"[SessionManager] Failed to cleanup session {sid}: {e}")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions (for debugging/admin)."""
        with self._lock:
            return [
                {
                    'session_id': s.session_id,
                    'created_at': s.created_at,
                    'last_active': s.last_active,
                    'state': s.generation_status.state,
                    'video_count': len(s.video_history)
                }
                for s in self._active_sessions.values()
            ]


class ProcessRegistry:
    """
    Session-aware subprocess registry.

    Replaces the global current_process variable with a session-keyed dictionary.
    Supports multiple concurrent sessions with independent process tracking.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._processes: Dict[str, Dict] = {}  # session_id -> process info

    def register(self, session_id: str, process: subprocess.Popen,
                 output_filename: str, params: Dict[str, Any] = None):
        """Register a new generation process for a session."""
        with self._lock:
            # If there's an existing process for this session, warn
            if session_id in self._processes:
                old_info = self._processes[session_id]
                old_process = old_info.get('process')
                if old_process and old_process.poll() is None:
                    print(f"[ProcessRegistry] Warning: Overwriting running process for session {session_id}")

            self._processes[session_id] = {
                'process': process,
                'output_filename': output_filename,
                'params': params or {},
                'started_at': time.time(),
                'stop_event': threading.Event()
            }
            print(f"[ProcessRegistry] Registered process for session {session_id}, PID: {process.pid}")

    def get(self, session_id: str) -> Optional[Dict]:
        """Get process info for a session."""
        with self._lock:
            return self._processes.get(session_id)

    def get_stop_event(self, session_id: str) -> Optional[threading.Event]:
        """Get the stop event for a session's process."""
        with self._lock:
            info = self._processes.get(session_id)
            if info:
                return info.get('stop_event')
            return None

    def get_output_filename(self, session_id: str) -> Optional[str]:
        """Get the output filename for a session's process."""
        with self._lock:
            info = self._processes.get(session_id)
            if info:
                return info.get('output_filename')
            return None

    def stop(self, session_id: str, timeout: float = 5.0) -> bool:
        """Stop the process for a session."""
        with self._lock:
            info = self._processes.get(session_id)
            if not info:
                return False

            # Set the stop event
            stop_event = info.get('stop_event')
            if stop_event:
                stop_event.set()

            process = info.get('process')
            if process and process.poll() is None:
                print(f"[ProcessRegistry] Stopping process for session {session_id}, PID: {process.pid}")
                process.terminate()
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    print(f"[ProcessRegistry] Force killing process for session {session_id}")
                    process.kill()
                    process.wait()

            return True

    def unregister(self, session_id: str):
        """Remove process from registry after completion."""
        with self._lock:
            if session_id in self._processes:
                del self._processes[session_id]
                print(f"[ProcessRegistry] Unregistered process for session {session_id}")

    def is_running(self, session_id: str) -> bool:
        """Check if a session has a running process."""
        with self._lock:
            info = self._processes.get(session_id)
            if not info:
                return False
            process = info.get('process')
            return process is not None and process.poll() is None

    def is_stopped(self, session_id: str) -> bool:
        """Check if stop was requested for a session."""
        with self._lock:
            info = self._processes.get(session_id)
            if not info:
                return False
            stop_event = info.get('stop_event')
            return stop_event is not None and stop_event.is_set()

    def get_all_running(self) -> List[str]:
        """Get list of session IDs with running processes."""
        with self._lock:
            return [
                sid for sid, info in self._processes.items()
                if info.get('process') and info['process'].poll() is None
            ]

    def cleanup_dead(self):
        """Remove entries for processes that have terminated."""
        with self._lock:
            to_remove = []
            for sid, info in self._processes.items():
                process = info.get('process')
                if process and process.poll() is not None:
                    to_remove.append(sid)

            for sid in to_remove:
                del self._processes[sid]
                print(f"[ProcessRegistry] Cleaned up dead process for session {sid}")


def write_progress_file(output_filename: str, progress_data: Dict[str, Any]):
    """
    Write progress to a sidecar file for recovery after disconnect.

    Progress file: {output_filename}.progress.json
    """
    if not output_filename:
        return

    progress_file = Path(output_filename).with_suffix('.progress.json')
    try:
        progress_data['timestamp'] = time.time()
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f)
    except Exception as e:
        print(f"[Progress] Failed to write progress file: {e}")


def read_progress_file(output_filename: str) -> Optional[Dict[str, Any]]:
    """Read progress from sidecar file."""
    if not output_filename:
        return None

    progress_file = Path(output_filename).with_suffix('.progress.json')
    try:
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[Progress] Failed to read progress file: {e}")
    return None


def cleanup_progress_file(output_filename: str):
    """Remove progress file after generation completes."""
    if not output_filename:
        return

    progress_file = Path(output_filename).with_suffix('.progress.json')
    try:
        if progress_file.exists():
            progress_file.unlink()
    except Exception:
        pass


# Global instances
session_manager = SessionManager()
process_registry = ProcessRegistry()


# Utility function to parse progress from line (duplicated here for self-containment)
def parse_progress_info(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse progress bar lines and extract structured information.
    Returns dict with step, total, percent, eta if found, else None.
    """
    import re
    line = line.strip()

    # Match TQDM format: "XX%|...|  current/total [...<HH:MM:SS"
    match = re.search(r'(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[.*?<([\d:]+)', line)
    if match:
        return {
            'percent': int(match.group(1)),
            'step': int(match.group(2)),
            'total': int(match.group(3)),
            'eta': match.group(4)
        }

    return None


# JavaScript for reconnect detection and session persistence
RECONNECT_JS = r"""
() => {
    console.log('[K1] Initializing session management...');

    const SESSION_KEY = 'k1_session_id';
    const PARAMS_KEY = 'k1_ui_params';

    // Get or create session ID
    let sessionId = localStorage.getItem(SESSION_KEY);

    // Store session ID in hidden input for Gradio
    function setSessionId(id) {
        const sessionInput = document.querySelector('#session_id_input input');
        if (sessionInput) {
            sessionInput.value = id;
            sessionInput.dispatchEvent(new Event('input', {bubbles: true}));
            console.log('[K1] Set session ID:', id);
        }
    }

    // Create reconnect banner
    function createReconnectBanner() {
        let banner = document.getElementById('k1-reconnect-banner');
        if (!banner) {
            banner = document.createElement('div');
            banner.id = 'k1-reconnect-banner';
            banner.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                background: linear-gradient(90deg, #ff6b6b, #ee5a5a);
                color: white;
                padding: 12px 20px;
                text-align: center;
                z-index: 99999;
                font-weight: bold;
                font-size: 14px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                display: none;
            `;
            banner.innerHTML = '&#9888; Connection lost. Attempting to reconnect...';
            document.body.prepend(banner);
        }
        return banner;
    }

    // Show/hide reconnect banner
    function showReconnectBanner() {
        const banner = createReconnectBanner();
        banner.style.display = 'block';
    }

    function hideReconnectBanner() {
        const banner = document.getElementById('k1-reconnect-banner');
        if (banner) {
            banner.style.display = 'none';
        }
    }

    // Create session status indicator
    function createSessionIndicator() {
        let indicator = document.getElementById('k1-session-indicator');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'k1-session-indicator';
            indicator.style.cssText = `
                position: fixed;
                bottom: 10px;
                right: 10px;
                background: rgba(0, 96, 223, 0.9);
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 11px;
                z-index: 9999;
                font-family: monospace;
            `;
            document.body.appendChild(indicator);
        }
        return indicator;
    }

    function updateSessionIndicator(id, status) {
        const indicator = createSessionIndicator();
        indicator.textContent = `Session: ${id} [${status}]`;
        indicator.style.background = status === 'connected' ? 'rgba(46, 204, 113, 0.9)' : 'rgba(255, 107, 107, 0.9)';
    }

    // Track WebSocket connection state
    let wsConnected = true;
    let reconnectAttempts = 0;

    // Intercept WebSocket to detect disconnections
    const OriginalWebSocket = window.WebSocket;
    window.WebSocket = function(url, protocols) {
        const ws = new OriginalWebSocket(url, protocols);

        ws.addEventListener('open', () => {
            if (!wsConnected) {
                console.log('[K1] WebSocket reconnected');
                reconnectAttempts = 0;
                hideReconnectBanner();
                updateSessionIndicator(sessionId || 'new', 'connected');

                // Trigger state recovery
                setTimeout(() => {
                    const recoveryBtn = document.querySelector('#recovery_trigger_btn');
                    if (recoveryBtn) {
                        console.log('[K1] Triggering state recovery...');
                        recoveryBtn.click();
                    }
                }, 500);
            }
            wsConnected = true;
        });

        ws.addEventListener('close', (event) => {
            wsConnected = false;
            reconnectAttempts++;
            console.log('[K1] WebSocket disconnected, attempt:', reconnectAttempts);
            showReconnectBanner();
            updateSessionIndicator(sessionId || 'unknown', 'disconnected');
        });

        ws.addEventListener('error', (error) => {
            console.error('[K1] WebSocket error:', error);
        });

        return ws;
    };

    // Set session ID if we have one from storage
    if (sessionId) {
        console.log('[K1] Found existing session:', sessionId);
        setTimeout(() => setSessionId(sessionId), 100);
    }

    // Update session indicator
    setTimeout(() => {
        updateSessionIndicator(sessionId || 'new', 'connected');
    }, 500);

    // Save session ID when it changes
    const observer = new MutationObserver((mutations) => {
        const sessionInput = document.querySelector('#session_id_input input');
        if (sessionInput && sessionInput.value && sessionInput.value !== sessionId) {
            sessionId = sessionInput.value;
            localStorage.setItem(SESSION_KEY, sessionId);
            console.log('[K1] Saved session ID:', sessionId);
            updateSessionIndicator(sessionId, 'connected');
        }
    });

    // Start observing after a delay to let Gradio initialize
    setTimeout(() => {
        const sessionContainer = document.querySelector('#session_id_input');
        if (sessionContainer) {
            observer.observe(sessionContainer, {
                subtree: true,
                childList: true,
                characterData: true,
                attributes: true
            });
        }
    }, 1000);

    // Periodic connection check
    setInterval(() => {
        if (!wsConnected && reconnectAttempts > 0) {
            showReconnectBanner();
        }
    }, 2000);

    console.log('[K1] Session management initialized');
}
"""


# API response helpers
def session_state_to_dict(session: SessionState) -> Dict[str, Any]:
    """Convert SessionState to JSON-serializable dict for API responses."""
    return {
        'session_id': session.session_id,
        'created_at': session.created_at,
        'last_active': session.last_active,
        'ui_params': session.ui_params,
        'generation_status': asdict(session.generation_status),
        'video_history': list(session.video_history),
        'active_tab': session.active_tab,
        'is_generating': process_registry.is_running(session.session_id)
    }
