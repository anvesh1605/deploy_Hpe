# Deployment Test Checklist

Use this checklist after backend and frontend deployment to verify the app works end to end.

## 1. Backend Health Check

- Open the backend URL in a browser or with `curl`.
- Check the health endpoint if available:

```bash
curl http://<backend-host>:<port>/api/health
```

- Check backend startup logs for errors.
- Confirm the service is listening on the expected `PORT`.

Example checks:

```bash
curl http://<backend-host>:<port>/
curl http://<backend-host>:<port>/api/health
```

## 2. Backend Environment Check

Verify the backend can access these environment variables:

- `MODEL_ROOT`
- `DATA_ROOT`
- `QWEN_MODEL_DIR`
- `RELEASE_LSTM_MODEL_DIR`
- `PRODUCT_LSTM_MODEL_DIR`
- `OLLAMA_BASE_URL`

Suggested check:

```bash
curl http://<backend-host>:<port>/api/health
```

Review the returned paths and connection status fields.

## 3. Data and Model Availability Check

Confirm the deployed server has access to:

- Release notes data
- Product documentation data
- Release LSTM model files
- Product LSTM model files
- Qwen or Ollama access, if required by the deployment

Checks to confirm:

- Lookup index files are present and readable.
- Model files are present and readable.
- The model service is reachable if Qwen/Ollama is required.

## 4. Frontend Check

- Open the deployed frontend in a browser.
- Confirm the frontend API URL points to the deployed backend.
- Open browser developer tools and confirm there are no CORS errors.
- Submit a question and confirm it reaches the backend.

Useful browser/network checks:

- The frontend should call the deployed backend URL, not `localhost`.
- The request should return a valid JSON response.

## 5. End-to-End Test Questions

Run these questions in the deployed UI:

### Release note test

`For 5420 AOS-CX 10.15.0001, when was version 10.15.1010 released?`

### Product syntax test

`For 6200 AOS-CX 10.06, what is the syntax of the IPv6 mld robustness command?`

### Product concept test

`For 10000 AOS-CX 10.07, explain the High Availability Overview.`

### Product command purpose test

`For 10000 AOS-CX 10.07, what does the redundancy switchover command do?`

## 6. Expected Behavior

- Backend returns `final_answer`.
- Frontend displays `final_answer`.
- LSTM predicts or routes the intent.
- Lookup provides the factual answer.
- Qwen explains the answer clearly without changing facts.
- Pure syntax answers must remain syntax-only and should not be reformatted into incorrect prose.

## 7. Common Deployment Errors

### CORS error

- Confirm the frontend backend URL is correct.
- Confirm the backend `ALLOWED_ORIGINS` setting includes the frontend origin.

### Frontend calling localhost instead of deployed backend

- Confirm `VITE_API_BASE_URL` or the runtime API base URL is set correctly.
- Rebuild or redeploy the frontend after changing the API URL.

### Backend cannot find data folder

- Confirm `DATA_ROOT` and the data subdirectory environment variables are set correctly.
- Confirm the server actually contains the copied data files.

### Backend cannot find model files

- Confirm `MODEL_ROOT` and the model directory environment variables are set correctly.
- Confirm the server actually contains the copied model files.

### Ollama/model service not reachable

- Confirm `OLLAMA_BASE_URL` is correct.
- Confirm the Ollama or model service is running and reachable from the backend server.

### Missing environment variables

- Confirm deployment environment variables are set in the hosting platform.
- Check backend health output for fallback paths being used.

### Port binding issue

- Confirm the hosting platform provides `PORT`, or use the platform default.
- Confirm the backend is listening on that port and no other service is using it.

## Final Verification

- Backend responds successfully.
- Frontend loads successfully.
- One release note question works.
- One product syntax question works.
- One product concept question works.
- One product command-purpose question works.

