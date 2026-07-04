# Live Demo Tunnel Setup

Use this guide when you want to expose the local app through a tunnel for a live demo.

## 1. Run the backend locally

- Start the backend on your machine.
- Keep the local data and model files on the machine.
- Do not push `.env`, data, or model files to GitHub.

## 2. Expose the backend with a tunnel

- Use `ngrok` or `cloudflared` to expose the backend port.
- Copy the public backend tunnel URL.

Example:

```bash
ngrok http 8000
```

or

```bash
cloudflared tunnel --url http://localhost:8000
```

## 3. Set the frontend API URL

- Set the frontend backend URL to the public backend tunnel URL.
- Use the environment variable or runtime config supported by the frontend.
- Do not hardcode the tunnel URL inside source code.

## 4. Run the frontend locally

- Start the frontend locally.
- Use host `0.0.0.0` if needed so the tunnel can reach it.

Example:

```bash
python -m http.server 5173
```

or your existing frontend dev/static server command.

## 5. Expose the frontend with a tunnel

- Use `ngrok` or `cloudflared` to expose the frontend port.
- Copy the public frontend tunnel URL.

Example:

```bash
ngrok http 5173
```

or

```bash
cloudflared tunnel --url http://localhost:5173
```

## 6. Share the frontend URL

- Use the frontend tunnel URL as the live demo link.
- The frontend will call the backend tunnel URL.

## Important Notes

- Data and models stay local on the machine or server.
- GitHub should contain code only.
- Tunnel URLs usually change each time unless you use a paid or static domain.
- Do not commit `.env`, data, or model files.

