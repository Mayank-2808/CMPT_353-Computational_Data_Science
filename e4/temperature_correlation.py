import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def haversine(lat1, lon1, lat2, lon2):
    
    R = 6371000  # Radius of Earth in meters
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance

def distance(city, stations):
   
    # Distances between city and all stations using haversine function
    city_coords = (city['latitude'], city['longitude'])
    station_coords = stations[['latitude', 'longitude']].values
    distances = np.apply_along_axis(lambda x: haversine(city_coords[0], city_coords[1], x[0], x[1]), axis=1, arr=station_coords)
    return distances

def best_tmax(city, stations):
    
    min_distance_index = np.argmin(distance(city, stations))
    
    return stations.iloc[min_distance_index]['avg_tmax'] / 10

def main(stations_file, city_data_file, output_file):
    
    stations = pd.read_json(stations_file, lines=True)
    cities = pd.read_csv(city_data_file)

    cities['area_km2'] = cities['area'] / 1e6

    # Remove outliers
    cities = cities.dropna(subset=['population', 'area_km2'])
    cities = cities[cities['area_km2'] <= 10000]

    cities['avg_tmax'] = cities.apply(best_tmax, stations=stations, axis=1)

    # Population density
    cities['population_density'] = cities['population'] / cities['area_km2']

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(cities['avg_tmax'], cities['population_density'], alpha=0.7)
    plt.title('Temperature vs Population Density')
    plt.xlabel('Avg Max Temperature (\u00b0C)')
    plt.ylabel('Population Density (people/km\u00b2)')
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    
    stations_file = sys.argv[1]
    city_data_file = sys.argv[2]
    output_file = sys.argv[3]

    main(stations_file, city_data_file, output_file)

# Distance function taken from my submission of e3
