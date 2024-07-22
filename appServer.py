from flask import Flask, request, jsonify
import numpy as np
import cv2
import insightface
import time
import requests
#from flash_cors import CORS

app = Flask(__name__)
#CORS(app)

# Global variables to keep track of entries and exits
entries = 0
exits = 0


@app.route('/detect', methods=['POST'])
def detect_person():
    global entries, exits

    data = request.get_json()
    action = data.get('action')
    timestamp = data.get('timestamp')
    obj = data.get('object')

    if action == 'enter':
        entries += 1
        print(f"Person entered at {timestamp}. Total entries: {entries}")
    elif action == 'exit':
        exits += 1
        print(f"Person exited at {timestamp}. Total exits: {exits}")
    elif action == 'danger':
        exits += 1
        print(f"Dangerous object '{obj}' detected at {timestamp}")

    return jsonify({"status": "success"}), 200


if __name__ == '__main__':
    #app.run()
    app.run(host='0.0.0.0', port=5000)
