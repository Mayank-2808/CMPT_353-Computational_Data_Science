import sys
import pandas as pd
import numpy as np
from xml.dom.minidom import getDOMImplementation
from xml.etree.ElementTree import parse
from pykalman import KalmanFilter

def output_gpx(points, output_filename):
    
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % pt['lat'])
        trkpt.setAttribute('lon', '%.8f' % pt['lon'])
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')

def haversine(lat1, lon1, lat2, lon2):
    
    R = 6371000  # Radius of Earth in meters
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance

def distance(df):
    
    latitudes = df['lat'].to_numpy()
    longitudes = df['lon'].to_numpy()
    
    distances = np.vectorize(haversine)(latitudes[:-1], longitudes[:-1], latitudes[1:], longitudes[1:])
    total_distance = np.sum(distances)
    return total_distance

def kalman_smooth(points, observation_covariance, transition_covariance):
    
    initial_state_mean = points.iloc[0].to_numpy().ravel()
    observations = points.to_numpy()
    
    kf = KalmanFilter(
        initial_state_mean=initial_state_mean,
        observation_covariance=observation_covariance,
        transition_matrices=np.eye(2),
        transition_covariance=transition_covariance,
        observation_matrices=np.eye(2),
    )

    smoothed_states, _ = kf.smooth(observations)

    smoothed_latitudes = smoothed_states[:, 0]
    smoothed_longitudes = smoothed_states[:, 1]

    smoothed_points = pd.DataFrame({'lat': smoothed_latitudes, 'lon': smoothed_longitudes})

    return smoothed_points

def get_data(gpx_file):
    
    tree = parse(gpx_file)
    root = tree.getroot()

    points = [{'lat': float(trkpt.get('lat')), 'lon': float(trkpt.get('lon'))} for trkpt in root.iter('{http://www.topografix.com/GPX/1/0}trkpt')]
    
    return pd.DataFrame(points)

def main():

    gpx_file = sys.argv[1]
    points = get_data(gpx_file)
    
    print('Unfiltered distance: %0.2f' % distance(points))
    
    observation_covariance = np.diag([1.4, 1.4])
    transition_covariance = np.diag([0.4, 0.4])

    smoothed_points = kalman_smooth(points, observation_covariance, transition_covariance)
    print('Filtered distance: %0.2f' % distance(smoothed_points))
    
    output_gpx(smoothed_points, 'out.gpx')

if __name__ == '__main__':
    main()

# Used ChatGPT to solve dimension conflicts in the kalman_smooth function
