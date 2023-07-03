from flask import Flask, render_template, request
import numpy as np

import pickle

model = pickle.load(open('flight_fare_model.pkl', 'rb'))
ler = pickle.load(open('flight_fare_label_encoder.pkl', 'rb'))
lea = pickle.load(open('flight_fare_label_encoder_air.pkl', 'rb'))
scaler = pickle.load(open('flight_fare_std.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def leroute(r):
    return ler.transform([r])[0]


def leair(airline):
    return lea.transform([airline])[0]


@app.route('/predict', methods=['POST'])
def prediction():
    duration = request.form.get('Duration')
    total_stops = request.form.get('total_stops')
    arrival = request.form.get('Arrival')
    depart = request.form.get('Departure')
    journey_date = request.form.get('journey_date')
    r1 = request.form.get('route1')
    r2 = request.form.get('route2')
    r3 = request.form.get('route3')
    r4 = request.form.get('route4')
    r5 = request.form.get('route5')
    airline = request.form.get('Airline')

    # preprocessing
    input_data = (leair(airline),
                  duration, total_stops,
                  arrival.split(':')[0], arrival.split(':')[1],
                  depart.split(':')[0], depart.split(':')[1],
                  journey_date.split('-')[1], journey_date.split('-')[2],
                  leroute(r1), leroute(r2), leroute(r3), leroute(r4), leroute(r5),
                  )

    # prediction
    std_data = scaler.transform(np.array(input_data).reshape(1, -1))
    result = model.predict(std_data)
    return "Price:   "+str(round(result[0], 2))


if __name__ == '__main__':
    app.run(debug=True)
