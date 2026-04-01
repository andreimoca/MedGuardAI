import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, User, Bot, AlertTriangle, Activity } from 'lucide-react';
import axios from 'axios';
import './index.css';

export default function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I am MedGuardAI. I can help answer medication questions based on FDA leaflets. Please fill in your profile on the left to get started.', status: 'success' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const [profile, setProfile] = useState({
    age: 30,
    weight: 70,
    allergies: '',
    conditions: ''
  });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);

    try {
      // Parse lists from strings
      const payload = {
        query: userMsg,
        patient_context: {
          age: parseInt(profile.age) || 0,
          weight: parseFloat(profile.weight) || 0,
          allergies: profile.allergies.split(',').map(s => s.trim()).filter(Boolean),
          conditions: profile.conditions.split(',').map(s => s.trim()).filter(Boolean)
        }
      };

      const res = await axios.post('http://localhost:8000/api/v1/ask', payload);

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: res.data.answer,
        status: res.data.status // 'success' or 'emergency'
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

          <div className="input-group">
            <label>Allergies (comma separated)</label>
            <input
              type="text"
              value={profile.allergies}
              onChange={e => setProfile({ ...profile, allergies: e.target.value })}
              placeholder="e.g. Penicillin, Peanuts"
            />
          </div>

          <div className="input-group">
            <label>Conditions (comma separated)</label>
            <input
              type="text"
              value={profile.conditions}
              onChange={e => setProfile({ ...profile, conditions: e.target.value })}
              placeholder="e.g. Asthma, Hypertension"
            />
          </div>
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
