# English to Nepali Translation Frontend

Single-page React client that sends English text to the Django translation API and renders the Nepali response.

## Features
- Minimal UI in [A3/translation-app/frontend/src/App.js](A3/translation-app/frontend/src/App.js) with textarea inputs and request button
- Styling in [A3/translation-app/frontend/src/App.css](A3/translation-app/frontend/src/App.css) for a centered card layout
- REST call to the backend POST /api/translate/ endpoint with loading state feedback
- Graceful error messaging if the backend is unreachable or returns an empty payload

## Prerequisites
- Node.js 18+ and npm
- Backend server running on http://localhost:8000 (see [A3/translation-app/backend](A3/translation-app/backend))

## Installation
```bash
npm install
```

## Development
```bash
npm start
```
- Create React App dev server runs on http://localhost:3000 with automatic reloads
- Proxy requests to the backend are handled in the fetch call inside [A3/translation-app/frontend/src/App.js](A3/translation-app/frontend/src/App.js)

## Building for Production
```bash
npm run build
```
- Outputs an optimized bundle in build/
- Deploy the build folder behind any static site host and set the backend URL via environment variables if needed

## Testing
```bash
npm test
```
- Runs react-scripts test runner with jest-dom matchers preconfigured via [A3/translation-app/frontend/src/setupTests.js](A3/translation-app/frontend/src/setupTests.js)

## Environment Configuration
- Update the fetch URL in [A3/translation-app/frontend/src/App.js](A3/translation-app/frontend/src/App.js) if the backend runs on a different host or port
- Ensure CORS is enabled on the Django backend when serving from a different origin

## Backend Setup (Django)
- Python 3.10+ with virtualenv support and pip installed
- From [A3/translation-app/backend](A3/translation-app/backend), create and activate a virtual environment, then install dependencies:
```bash
python -m venv env
env\\Scripts\\activate
pip install -r requirements.txt
```
- Apply database migrations and start the API server:
```bash
python manage.py migrate
python manage.py runserver
```
- The translation endpoint is defined in [A3/translation-app/backend/api/views.py](A3/translation-app/backend/api/views.py) and exposed via [A3/translation-app/backend/api/urls.py](A3/translation-app/backend/api/urls.py)
- Default server listens on http://localhost:8000 and serves POST /api/translate/ expecting `{ "text": "Your sentence" }`

## API Responses
- Success: `{ "translation": "<Nepali text>", "debug_tokens": "<token sequence>", "input_tokens": ["<sos>", ...] }`
- Error: `{"error": "message"}` with appropriate HTTP status (400 for missing text, 500 for unexpected errors)

## Project Structure
- [A3/translation-app/frontend/public](A3/translation-app/frontend/public) contains the HTML template and static assets
- [A3/translation-app/frontend/src/index.js](A3/translation-app/frontend/src/index.js) boots the React tree and wiring for reportWebVitals
- [A3/translation-app/frontend/src/reportWebVitals.js](A3/translation-app/frontend/src/reportWebVitals.js) is optional performance logging
- [A3/translation-app/frontend/src/App.test.js](A3/translation-app/frontend/src/App.test.js) shows a smoke test scaffold

## Troubleshooting
- If fetch requests fail, confirm the backend runserver log shows POST /api/translate/ and CORS headers
- When builds miss CSS changes, clear the build/ directory and rerun npm run build
- For dependency issues, remove node_modules/ and package-lock.json then reinstall
