# Two-Laptop Demo Checklist

**Laptop 1** runs the API (model, HMM, evaluator trips, ground truth).
**Laptop 2** runs the UI (no data, no model; it only talks to the API).

## Days before: rehearse once with two real laptops

- [ ] Both laptops: `git pull` and the same commit checked out (`git log -1`)
- [ ] Both laptops: `pip install -r requirements-lock.txt` (same versions the models were trained with)
- [ ] Both laptops: `python -c "import sklearn; print(sklearn.__version__)"` prints `1.7.2`
- [ ] Laptop 1: `models/final_model.joblib` present (or `logreg.joblib` as fallback)
- [ ] Laptop 1: right-click `setup_firewall.bat` → **Run as administrator** (one time)
- [ ] Full run-through on a **phone hotspot**, with all 10 student_test trips clicked one by one

## On the day

1. Turn on a phone hotspot and connect **both** laptops to it.
   Campus Wi-Fi often blocks laptop-to-laptop traffic.
2. **Laptop 1:** copy the organisers' evaluator trips to a folder, then run
   ```
   run_api.bat C:\path\to\evaluator_trips
   ```
   Note the IPv4 address printed next to **Wi-Fi** at the top (e.g. `192.168.43.10`).
   Ignore VirtualBox / vEthernet adapters (e.g. `192.168.56.1`), because laptop 2 can't reach those.
   Keep this window visible: the evaluator must see the request logs here.
3. **Laptop 2:** run
   ```
   run_ui.bat 192.168.43.10
   ```
   The browser opens. Click **Test connection** in the sidebar. It should say "Connected".
4. Check that the trip dropdown shows the **10 evaluator trips**.
5. Evaluator picks a trip → **Reconstruct route** → metrics and map appear.
   On Laptop 1 you should see lines like
   `POST /predict from 192.168.43.11 -> 200` and `trip XXXXX: ... HMM top-1 ...`.
6. Take screenshots: the UI with results, and the Laptop 1 terminal with the logs (needed for the report).

## If something goes wrong

| Symptom | Fix |
|---|---|
| UI: "Cannot reach API" | Same hotspot? Right IP (`ipconfig` on Laptop 1)? Firewall script run as admin? From Laptop 2, open `http://<ip>:8000/health` in a browser |
| `/health` works on Laptop 1 (`127.0.0.1`) but not from Laptop 2 | Firewall or network isolation. Re-run `setup_firewall.bat` as admin, or switch to the hotspot |
| API fails to start: model load error | Library version mismatch. Run `pip install -r requirements-lock.txt` on Laptop 1 |
| Map has lines but no street background | Laptop 2 has no internet (the map tiles are online). The routes are still correct; turn on mobile data on the hotspot |
| Metrics say "not available" | The evaluator folder has no `.route` files. Ask the organisers where the reference routes are |
| Port 8000 already in use | Close the old API window, or `netstat -ano \| findstr :8000` then `taskkill /PID <pid> /F` |

## Rules to respect during the demo

- The UI must go through the API. No pre-computed or hardcoded outputs.
- Ground truth stays on Laptop 1. The UI only receives predictions and metrics.
- Do not train or tune anything on the evaluator trips.
