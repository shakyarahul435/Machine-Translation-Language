import React, { useState } from 'react';
import './App.css';

function App() {
  const [englishText, setEnglishText] = useState('');
  const [nepaliText, setNepaliText] = useState('');
  const [loading, setLoading] = useState(false);

  const handleTranslate = async () => {
    if (!englishText.trim()) return;
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/translate/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: englishText }),
      });
      const data = await response.json();
      setNepaliText(data.translation || 'Translation failed');
    } catch (error) {
      setNepaliText('Error: Unable to connect to backend');
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>English to Nepali Translator</h1>
        <div className="translator-container">
          <div className="input-section">
            <label htmlFor="english-input">English Text:</label>
            <textarea
              id="english-input"
              value={englishText}
              onChange={(e) => setEnglishText(e.target.value)}
              placeholder="Enter English text here..."
              rows="4"
            />
          </div>
          <button onClick={handleTranslate} disabled={loading}>
            {loading ? 'Translating...' : 'Translate'}
          </button>
          <div className="output-section">
            <label htmlFor="nepali-output">Nepali Translation:</label>
            <textarea
              id="nepali-output"
              value={nepaliText}
              readOnly
              placeholder="Translation will appear here..."
              rows="4"
            />
          </div>
        </div>
      </header>
    </div>
  );
}

export default App;
