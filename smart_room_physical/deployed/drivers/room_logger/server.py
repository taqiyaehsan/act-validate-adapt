import os
import csv
import time
from datetime import datetime, timezone #for utc timestamp!

from flask import Flask, render_template, request, jsonify, redirect, url_for
from waitress import serve


#-----------------Config---------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))

CSV_PATH = os.path.join(DATA_DIR, "window_log.csv")
LOGGED_BY = "manual_app"
MAX_NOTES = 200 #max note length 
HISTORY_N = 20 #num notes to display in history section


app = Flask(__name__)



#-----------------Utility functions-----------------------------------------------------

def utc_iso_z() -> str:
    """Returns the current UTC time in ISO 8601 format with 'Z' suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
# above^ current time utc; remove microseconds; iso string; timezone offset Z


def ensure_data_file():
    """ Ensures the data directory and csv file exist. if they do not, then create them."""
    os.makedirs(DATA_DIR, exist_ok=True)#create /data directory if missing

    if not os.path.exists(CSV_PATH): 
        #if csv file does not exist, create with header row
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvfile: #create new file
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'window_state', 'occupancy','logged_by', 'notes']) #write header row

def sanitize_notes(notes:str) -> str:
    """cleans notes to prevent formatting issues in csv"""
    if not notes:
        return ""
    
    notes = notes.strip() #remove leading/trailing whitespace
    
    #to prevent multi-line issues:
    notes = notes.replace("\n", " ").replace("\r", " ") 

    return notes[:MAX_NOTES] #length limit


class FileLock:
    """linux file lock using fcntl. """
    def __init__(self, fp):
        self.fp = fp  # store file pointer

    def __enter__(self):
        import fcntl
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)  
        # blocks other writers
        return self.fp

    def __exit__(self, exc_type, exc, tb):
        import fcntl
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)  
        # unlock

def append_row(row):
    """appends single row to csv file"""
    ensure_data_file()

    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
        with FileLock(csvfile):  # get lock for writing
            writer = csv.writer(csvfile)
            writer.writerow(row)  # write the row
            csvfile.flush()
            os.fsync(csvfile.fileno())  # data is written to disk


def read_last_n(n: int):
    """reads last n rows from csv file"""
    ensure_data_file()

    rows = []

    with open(CSV_PATH, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    return rows[-n:][::-1]  # return last n rows in reverse order (newest first)


def get_current_state():
    """returns current window state based on last entry in csv file"""
    rows=read_last_n(1) #get most recent row
    if not rows: return None
    
    row = rows[0]
    return{"timestamp": row["timestamp"],
           "window_state": int(row["window_state"]),
           "occupancy": int(row.get("occupancy", 0)),
           "logged_by": row["logged_by"],
           "notes": row["notes"]}


#-----------------routes-----------------------------------------------------
@app.route('/')
@app.route('/index')
def index():
    current=get_current_state()#self explanatory
    history=read_last_n(HISTORY_N) 
    state_text='Unknown'
    state_time='--'

    if current:
        state_text="Open" if current["window_state"]==1 else "Closed"
        state_time=current["timestamp"]
    return render_template('index.html',
                           current_state_text=state_text,
                           current_state_time=state_time,
                           history=history,
                           message=None,
                           ok=True)


@app.route('/health')
def health():
    # simple health check endpoint
    try:
        ensure_data_file()  # check if we can access the data file
        return jsonify({'status':'ok'}), 200 #if we can access the file, we're healthy
    except Exception as e:
        return jsonify({'status':'error', 'error': str(e)}), 500 #if there's an error accessing the file, return error status
    

@app.route('/get_history')
def get_history(): #return last 20 entries as json
    try:
        return jsonify(read_last_n(HISTORY_N)), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route("/get_current_state")
def get_current_state_endpoint():
    # returning most recent state
    current = get_current_state()
    if not current:
        return jsonify({"state": "unknown"}), 200
    return jsonify(current), 200


@app.route("/log_window", methods=["POST"])
def log_window():
    """
    Logs a new window state.
    Accepts form or JSON.
    """

    # Server-side double tap protection
    now = time.time()
    last = getattr(app, "_last_log_ts", 0.0)

    if now - last < 0.7:
        return jsonify({"error": "Too fast"}), 429

    # Determine if request is JSON or form
    if request.is_json:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON"}), 400

        window_state = data.get("window_state")
        occupancy = data.get("occupancy")
        notes = data.get("notes", "")
    else:
        window_state = request.form.get("window_state")
        occupancy = request.form.get("occupancy")
        notes = request.form.get("notes", "")

    

    # Validate window_state
    try:
        window_state = int(window_state)
    except:
        return jsonify({"error": "window_state must be 0 or 1"}), 400

    if window_state not in (0, 1):
        return jsonify({"error": "window_state must be 0 or 1"}), 400


    #validating room occupancy
    try:
        occupancy = int(occupancy)
    except:
        return jsonify({"error": "occupancy must be 0 or 1"}), 400

    if occupancy not in (0, 1):
        return jsonify({"error": "occupancy must be 0 or 1"}), 400


    # Sanitize notes
    notes = sanitize_notes(notes)

    # Create timestamp
    timestamp = utc_iso_z()

    append_row([timestamp, window_state, occupancy, LOGGED_BY, notes])
        # ypdate last log time
    app._last_log_ts = now

    if not request.is_json:
        return redirect(url_for("index"))

   


    return jsonify({
    "ok": True,
    "timestamp": timestamp,
    "window_state": window_state,
    "occupancy": occupancy
}), 200



 
#-----------startup------------------------------------------------------

if __name__ == '__main__':
    ensure_data_file()
    serve(app, host="192.168.0.208", port=int(os.environ.get("PORT", 5000)))

