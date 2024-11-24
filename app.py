from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from neural_networks import visualize

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    activation = request.json['activation']
    lr = float(request.json['lr'])
    step_num = int(request.json['step_num'])

    visualize(activation, lr, step_num)
    result_gif = "results/visualize.gif"

    return jsonify({"result_gif": result_gif if os.path.exists(result_gif) else None})

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
