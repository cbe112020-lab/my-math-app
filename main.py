from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允許跨網域存取

@app.route('/api/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    x = float(data.get('number', 0))

    # 計算 x * 10 + 2
    final_result = x * 10 + 2

    return jsonify({'status': 'success', 'result': final_result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)