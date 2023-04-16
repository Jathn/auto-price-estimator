import gradio as gr
import pandas as pd
import pickle

with open('app_pipeline.pkl', 'rb') as f:
    pipe = pickle.load(f)

state_coords = {
    'Alabama': [32.3182, -86.9023],
    'Alaska': [66.1605, -153.3691],
    'Arizona': [33.7298, -111.4312],
    'Arkansas': [34.9697, -92.3731],
    'California': [36.1162, -119.6816],
    'Colorado': [39.0598, -105.3111],
    'Connecticut': [41.5978, -72.7553],
    'Delaware': [39.3185, -75.5071],
    'Florida': [27.7663, -81.6868],
    'Georgia': [33.0406, -83.6431],
    'Hawaii': [21.0943, -157.4983],
    'Idaho': [44.2405, -114.4788],
    'Illinois': [40.3495, -88.9861],
    'Indiana': [39.8494, -86.2583],
    'Iowa': [42.0115, -93.2105],
    'Kansas': [38.5266, -96.7265],
    'Kentucky': [37.6681, -84.6701],
    'Louisiana': [31.1695, -91.8678],
    'Maine': [44.6939, -69.3819],
    'Maryland': [39.0639, -76.8021],
    'Massachusetts': [42.2302, -71.5301],
    'Michigan': [43.3266, -84.5361],
    'Minnesota': [45.6945, -93.9002],
    'Mississippi': [32.3546689, -89.3985283],
    'Missouri': [38.462853, -92.302],
    'Montana': [46.9652606, -109.533691],
    'Nebraska': [41.1253705, -98.2680826],
    'Nevada': [38.8026097, -116.419389],
    'New Hampshire': [43.1938516, -71.5723953],
    'New Jersey': [40.0583238, -74.4056612],
    'New Mexico': [34.3071446, -106.0180674],
    'New York': [42.1657266, -74.9480517],
    'North Carolina': [35.6300666, -79.8064195],
    'North Dakota': [47.5289124, -99.7840122],
    'Ohio': [40.3887839, -82.7649153],
    'Oklahoma': [35.5653429, -96.9289175],
    'Oregon': [44.5720216, -122.0709384],
    'Pennsylvania': [40.5907529, -77.2097553],
    'Rhode Island': [41.6809701, -71.5117807],
    'South Carolina': [33.8568928, -80.9450078],
    'South Dakota': [44.2997826, -99.4388286],
    'Tennessee': [35.7478456, -86.6923451],
    'Texas': [31.0544878, -97.5634611],
    'Utah': [39.3209801, -111.0937311],
    'Vermont': [44.5588028, -72.5778415],
    'Virginia': [37.9268684, -78.0249026],
    'Washington': [47.7510741, -120.7401385],
    'West Virginia': [38.5976262, -80.4549026],
    'Wisconsin': [44.7862968, -89.8267049],
    'Wyoming': [43.000325, -107.5545669]
}


def predict_price(year, manufacturer, type, odometer, paint_color, fuel, state):
    # preprocess inputs
    input_df = pd.DataFrame({
        'year': [year],
        'manufacturer': [manufacturer],
        'type': [type],
        'odometer': [odometer],
        'paint_color': [paint_color],
        'fuel': [fuel],
        'lat':state_coords[state][0],
        'long':state_coords[state][1]
    })

    # make prediction
    prediction = pipe.predict(input_df)[0]

    # return result
    return round(prediction, 2)

iface = gr.Interface(fn=predict_price, 
                     inputs=["number", "text", "text", "number", "text", "text", "text"], 
                     outputs="number",
                     title="Auto Price Estimator",
                     description="Welcome to the Auto Price Estimator!\nSimply enter some details about the car you’re interested in and get an estimated price. The model uses machine learning to analyze the input data and provide you with the most accurate prediction possible. Input the year, manufacturer, type, odometer, paint color, and fuel type to get started.\nInput details:\n- write states with upper and lower case, example: New York.\n- Manufacturer names are all in lower, example: ford.\n- The different types are: SUV, bus, convertible, coupe, hatchback, mini-van, offroad, other, pickup, sedan, truck, van, wagon.\n- Odometer (miles)\n- paint_color: lower case\n- fuel: diesel, gas, hybrid, electric, other\n- State write with upper and lower case as in the example: New York.\nThe model has an accuracy of 84% and a mean absolute error of 1600.\nLink to github repoitory: NaN")

iface.launch()