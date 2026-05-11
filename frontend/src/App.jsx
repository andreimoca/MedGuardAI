import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, User, Bot, AlertTriangle, Activity, Plus, X, RefreshCw, ThumbsUp, ThumbsDown } from 'lucide-react';
import axios from 'axios';
import './index.css';

const API_BASE = 'http://localhost:8000';

const WELCOME_MESSAGE = {
  role: 'assistant',
  content: 'Hello! I am MedGuardAI. I can help answer medication questions based on FDA leaflets. Please fill in your profile on the left to get started.',
  status: 'success'
};

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
    <div className="input-group">
      <label>{label}</label>
      {profile[field].length > 0 && (
        <div className="chip-list">
          {profile[field].map((item, idx) => (
            <span key={item} className="chip">
              {item}
              <button type="button" onClick={() => removeItem(field, idx)} aria-label={`Remove ${item}`}>
                <X size={12} />
              </button>
            </span>
          ))}
        </div>
      )}
      <div className="chip-input-row">
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
          className="chip-add-btn"
          onClick={() => addItem(field, value, setValue)}
          disabled={!value.trim()}
          aria-label={`Add ${label.toLowerCase()}`}
        >
          <Plus size={14} />
        </button>
      </div>
    </div>
  );

  return (
    <div className="app-container">
      {/* Sidebar Profile */}
      <motion.aside
        initial={{ x: -50, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="sidebar"
      >
        <div className="brand">
          <Activity size={28} color="#818cf8" />
          MedGuardAI
        </div>

        <div className="profile-container">
          <div className="profile-heading">Patient Profile</div>

          <div className="input-group">
            <label>Age (years)</label>
            <input
              type="number"
              value={profile.age}
              onChange={e => setProfile({ ...profile, age: e.target.value })}
              min="0"
            />
          </div>

          <div className="input-group">
            <label>Weight (kg)</label>
            <input
              type="number"
              value={profile.weight}
              onChange={e => setProfile({ ...profile, weight: e.target.value })}
              min="0"
            />
          </div>

          {renderChipInput('Allergies', 'allergies', allergyInput, setAllergyInput, 'e.g. Penicillin, Peanuts…')}
          {renderChipInput('Conditions', 'conditions', conditionInput, setConditionInput, 'e.g. Asthma, Hypertension…')}
        </div>

        <div style={{ marginTop: 'auto', fontSize: '0.8rem', color: '#64748b' }}>
          By using this service, you agree to context-aware Agentic RAG evaluation. In an emergency, always dial your local emergency number.
        </div>
      </motion.aside>

      {/* Main Chat Area */}
      <motion.main
        initial={{ y: 50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="main-chat"
      >
        <div className="chat-header">
          <span className="chat-header-title">MedGuardAI Assistant</span>
          <button className="new-chat-btn" onClick={handleNewConversation} title="Start a new conversation">
            <RefreshCw size={16} /> New conversation
          </button>
        </div>

        <div className="chat-messages">
          <AnimatePresence>
            {messages.map((m, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, scale: 0.95, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className={`message ${m.role} ${m.status === 'emergency' ? 'emergency' : ''}`}
              >
                <div className="avatar">
                  {m.role === 'user' ? <User size={20} /> : (m.status === 'emergency' ? <AlertTriangle size={20} /> : <Bot size={20} />)}
                </div>
                <div className="message-bubble">
                  {m.content}

                  {m.role === 'assistant' && m.feedback !== undefined && m.status === 'success' && (
                    <div className="feedback-row">
                      {m.feedback === 'sent' ? (
                        <span className="feedback-thanks">Thanks for the feedback ✓</span>
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
                            <ThumbsUp size={14} />
                          </button>
                          <button
                            type="button"
                            className="feedback-btn"
                            onClick={() => sendFeedback(idx, 'down')}
                            aria-label="Not helpful"
                            title="Not helpful"
                          >
                            <ThumbsDown size={14} />
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
                className="message assistant"
              >
                <div className="avatar"><Bot size={20} /></div>
                <div className="message-bubble typing-indicator">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-container">
          <form onSubmit={handleSend} className="chat-input-wrapper">
            <input
              type="text"
              className="chat-input"
              placeholder="Ask about a medication, dosage, or symptom..."
              value={input}
              onChange={e => setInput(e.target.value)}
              disabled={loading}
            />
            <button type="submit" className="chat-submit" disabled={!input.trim() || loading}>
              <Send size={18} />
            </button>
          </form>
        </div>
      </motion.main>
    </div>
  );
}
