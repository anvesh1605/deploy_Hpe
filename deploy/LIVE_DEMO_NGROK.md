# Ngrok Live Demo Setup

Use this setup for a temporary live demo with ngrok.

## Step 1: Run backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Step 2: Start backend ngrok

```bash
ngrok http 8000
```

## Step 3: Copy backend HTTPS URL

- Copy the HTTPS forwarding URL shown by ngrok for the backend.

## Step 4: Create or update frontend `.env.local`

Set the backend URL in the frontend environment file:

```env
VITE_API_BASE_URL=<backend-ngrok-url>
```

## Step 5: Run frontend

If your frontend is Vite-based:

```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

If you are using the current static frontend setup, keep the local frontend server you already use and point it at the backend ngrok URL through runtime config.

## Step 6: Start frontend ngrok

```bash
ngrok http 5173
```

If your frontend is served on a different local port, expose that port instead.

## Step 7: Share only the frontend HTTPS URL

- Share the frontend ngrok URL with your audience.
- The frontend should call the backend ngrok URL.

## Important Notes

- Keep the laptop on for the whole demo.
- Keep the backend terminal running.
- Keep the backend ngrok terminal running.
- Keep the frontend terminal running.
- Keep the frontend ngrok terminal running.
- Free ngrok URLs change every restart unless you use a paid or static domain.
- Data and models stay local.
- GitHub contains code only.
- Do not commit `.env`, data, or model files.

