import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send, User, Bot, AlertTriangle, HeartPulse, Plus, X, RefreshCw,
  ThumbsUp, ThumbsDown, ShieldCheck, ClipboardList, CheckCircle2, Stethoscope,
} from 'lucide-react';
import axios from 'axios';
import './index.css';

const API_BASE = 'http://localhost:8000';

const WELCOME_MESSAGE = {
  role: 'assistant',
  content: 'Hello — I\'m MedGuardAI. I answer medication questions grounded in FDA leaflets. Fill in the patient profile on the left so I can tailor my guidance, then ask me anything.',
  status: 'success'
};

/* Decorative ECG / heartbeat glyph used in the brand mark */
function HeartbeatGlyph({ size = 22 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M2 12.5h3.4l1.7-4.2a.6.6 0 0 1 1.12-.03L11 16.2a.6.6 0 0 0 1.12.06L14.2 9a.6.6 0 0 1 1.1-.05l1.5 3.55H22"
        stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round"
      />
    </svg>
  );
}

export default function App() {
  const [messages, setMessages] = useState([WELCOME_MESSAGE]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const [profile, setProfile] = useState({
    age: 30,
    weight: 70,
    allergies: [],
    conditions: []
  });
  const [allergyInput, setAllergyInput] = useState('');
  const [conditionInput, setConditionInput] = useState('');

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const addItem = (field, value, clearFn) => {
    const trimmed = value.trim();
    if (!trimmed) return;
    setProfile(prev => (
      prev[field].includes(trimmed)
        ? prev
        : { ...prev, [field]: [...prev[field], trimmed] }
    ));
    clearFn('');
  };

  const removeItem = (field, idx) => {
    setProfile(prev => ({ ...prev, [field]: prev[field].filter((_, i) => i !== idx) }));
  };

  const handleNewConversation = () => {
    setMessages([WELCOME_MESSAGE]);
    setInput('');
    setLoading(false);
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);

    try {
      const payload = {
        query: userMsg,
        patient_context: {
          age: parseInt(profile.age) || 0,
          weight: parseFloat(profile.weight) || 0,
          allergies: profile.allergies,
          conditions: profile.conditions
        }
      };

      const res = await axios.post(`${API_BASE}/api/v1/ask`, payload);

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: res.data.answer,
        status: res.data.status, // 'success' or 'emergency'
        forQuery: userMsg,
        forContext: payload.patient_context,
        feedback: null // null | 'sent'
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error connecting to API. Is the backend running? (${err.message})`,
        status: 'error'
      }]);
    } finally {
      setLoading(false);
    }
  };

  const sendFeedback = async (idx, rating) => {
    const m = messages[idx];
    if (!m || m.role !== 'assistant') return;
    try {
      await axios.post(`${API_BASE}/api/v1/feedback`, {
        query: m.forQuery,
        patient_context: m.forContext,
        answer: m.content,
        rating,
        status: m.status
      });
      setMessages(prev => prev.map((x, i) => (i === idx ? { ...x, feedback: 'sent' } : x)));
    } catch (err) {
      // Non-fatal: just log; the chat keeps working without feedback persistence.
      console.warn('Could not save feedback:', err.message);
    }
  };

  const renderChipInput = (label, field, value, setValue, placeholder) => (
    <div className="field">
      <label>{label}</label>
      {profile[field].length > 0 && (
        <div className="chips">
          {profile[field].map((item, idx) => (
            <span key={item} className="chip">
              {item}
              <button type="button" onClick={() => removeItem(field, idx)} aria-label={`Remove ${item}`}>
                <X size={11} strokeWidth={2.6} />
              </button>
            </span>
          ))}
        </div>
      )}
      <div className="chip-row">
        <input
          type="text"
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter') {
              e.preventDefault();
              addItem(field, value, setValue);
            }
          }}
          placeholder={placeholder}
        />
        <button
          type="button"
          className="chip-add"
          onClick={() => addItem(field, value, setValue)}
          disabled={!value.trim()}
          aria-label={`Add ${label.toLowerCase()}`}
        >
          <Plus size={15} strokeWidth={2.4} />
        </button>
      </div>
    </div>
  );

  return (
    <div className="app">
      <div className="app-bg" aria-hidden="true" />

      <div className="app-shell">
        {/* Sidebar — patient profile */}
        <motion.aside
          initial={{ x: -24, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="panel sidebar"
        >
          <div className="brand">
            <div className="mark"><HeartbeatGlyph size={22} /></div>
            <div className="brand-text">
              <span className="brand-name">Med<b>Guard</b>AI</span>
              <span className="brand-sub">Medication safety assistant</span>
            </div>
          </div>

          <div className="profile">
            <div className="profile-title">
              <ClipboardList size={17} strokeWidth={2.2} />
              Patient profile
            </div>

            <div className="field-row">
              <div className="field">
                <label>Age · years</label>
                <input
                  type="number"
                  value={profile.age}
                  onChange={e => setProfile({ ...profile, age: e.target.value })}
                  min="0"
                />
              </div>
              <div className="field">
                <label>Weight · kg</label>
                <input
                  type="number"
                  value={profile.weight}
                  onChange={e => setProfile({ ...profile, weight: e.target.value })}
                  min="0"
                />
              </div>
            </div>

            {renderChipInput('Allergies', 'allergies', allergyInput, setAllergyInput, 'e.g. Penicillin, peanuts…')}
            {renderChipInput('Conditions', 'conditions', conditionInput, setConditionInput, 'e.g. Asthma, hypertension…')}

            <div className="trust">
              <ShieldCheck size={16} strokeWidth={2.2} />
              Answers grounded in FDA drug leaflets
            </div>
          </div>

          <div className="disclaimer">
            By using this service you agree to context-aware Agentic RAG evaluation.
            This tool does not replace professional medical advice — <b>in an emergency, call your local emergency number.</b>
          </div>
        </motion.aside>

        {/* Main — conversation */}
        <motion.main
          initial={{ y: 24, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.12, ease: [0.22, 1, 0.36, 1] }}
          className="panel chat"
        >
          <div className="chat-bar">
            <div className="chat-id">
              <div className="pulse-badge"><HeartPulse size={18} strokeWidth={2.2} /></div>
              <div>
                <h1>MedGuardAI Assistant</h1>
                <div className="status"><span className="dot" /> Online · ready to help</div>
              </div>
            </div>
            <button className="new-chat" onClick={handleNewConversation} title="Start a new conversation">
              <RefreshCw size={15} strokeWidth={2.2} /> New conversation
            </button>
          </div>

          <div className="messages">
            <AnimatePresence initial={false}>
              {messages.map((m, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 12, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
                  className={`msg ${m.role} ${m.status === 'emergency' ? 'emergency' : ''}`}
                >
                  <div className="msg-avatar">
                    {m.role === 'user'
                      ? <User size={18} strokeWidth={2.2} />
                      : (m.status === 'emergency'
                          ? <AlertTriangle size={18} strokeWidth={2.2} />
                          : <Stethoscope size={18} strokeWidth={2.2} />)}
                  </div>
                  <div className="bubble">
                    {m.content}

                    {m.role === 'assistant' && m.feedback !== undefined && m.status === 'success' && (
                      <div className="feedback">
                        {m.feedback === 'sent' ? (
                          <span className="feedback-thanks">
                            <CheckCircle2 size={14} strokeWidth={2.4} /> Thanks for the feedback
                          </span>
                        ) : (
                          <>
                            <span className="feedback-label">Was this helpful?</span>
                            <button
                              type="button"
                              className="feedback-btn"
                              onClick={() => sendFeedback(idx, 'up')}
                              aria-label="Helpful"
                              title="Helpful"
                            >
                              <ThumbsUp size={14} strokeWidth={2.2} />
                            </button>
                            <button
                              type="button"
                              className="feedback-btn"
                              onClick={() => sendFeedback(idx, 'down')}
                              aria-label="Not helpful"
                              title="Not helpful"
                            >
                              <ThumbsDown size={14} strokeWidth={2.2} />
                            </button>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}

              {loading && (
                <motion.div
                  key="loading-indicator"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="msg assistant"
                >
                  <div className="msg-avatar"><Bot size={18} strokeWidth={2.2} /></div>
                  <div className="bubble">
                    <div className="typing"><span /><span /><span /></div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>

          <div className="composer">
            <form onSubmit={handleSend}>
              <input
                type="text"
                placeholder="Ask about a medication, dosage, or symptom…"
                value={input}
                onChange={e => setInput(e.target.value)}
                disabled={loading}
              />
              <button type="submit" className="send-btn" disabled={!input.trim() || loading} aria-label="Send message">
                <Send size={18} strokeWidth={2.2} />
              </button>
            </form>
            <p className="composer-hint">
              MedGuardAI can make mistakes — always confirm with a <b>licensed healthcare professional</b>.
            </p>
          </div>
        </motion.main>
      </div>
    </div>
  );
}
