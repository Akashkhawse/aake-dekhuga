# SmartAI Dashboard (Flask)

## Quick run (local)
1. Create and activate virtualenv
   python3 -m venv venv
   source venv/bin/activate

2. Install
   pip install -r requirements.txt

3. Create `.env` with PERPLEXITY_API_KEY or OPENAI_API_KEY if you want AI replies.
   (If no keys provided, assistant endpoint will return helpful message.)

4. Run locally:
   python app.py
   - open http://127.0.0.1:5000

5. If camera not working (macOS permission or server), set `DISABLE_CAMERA=1` in .env.

## Deploy (Render / Heroku)
- Commit repo to GitHub.
- On Render or Heroku create a new web service and connect repo.
- Ensure build command `pip install -r requirements.txt` and start command from `Procfile`.
- Add environment variables in platform dashboard (PERPLEXITY_API_KEY, OPENAI_API_KEY, DISABLE_CAMERA).

## Notes / Troubleshooting
- macOS microphone/camera access: allow in System Settings -> Privacy & Security.
- If `pyaudio` fails to install: install portaudio via Homebrew: `brew install portaudio` then `pip install pyaudio`.
- On server/cloud you likely don't have physical camera — set `DISABLE_CAMERA=1`.
# aake-dekhuga
# aake-dekhuga
# Self-monitoring-with-AI
# newproject
