# Member 4: evaluator UI (laptop 2): lists the evaluation trips, sends the chosen trip to the API and shows the result
import os
import folium
import requests
import streamlit as st

st.set_page_config(page_title='PathSnap', layout='wide')
st.title('PathSnap: map matching and route reconstruction')

# address of the API laptop, e.g. http://192.168.1.20:8000
api_url = st.sidebar.text_input('API address (laptop 1)', os.environ.get('PATHSNAP_API', 'http://localhost:8000'))

try:
    listing = requests.get(f"{api_url}/trips", timeout=10).json()
except requests.RequestException as e:
    st.error(f"Cannot reach the API at {api_url}: {e}")
    st.stop()

trip_id = st.selectbox(f"Evaluation trip ({len(listing['trips'])} in {listing['data_dir']})", listing['trips'])

if st.button('Run map matching', type='primary'):
    with st.spinner(f"Laptop 1 is processing trip {trip_id}..."):
        response = requests.post(f"{api_url}/predict/{trip_id}", timeout=600)
    if response.ok:
        st.session_state['result'] = response.json()
    else:
        st.error(f"API error {response.status_code}: {response.text}")

result = st.session_state.get('result')
if result and result['trip_id'] == trip_id:
    scores = result['scores']
    if scores:
        c1, c2, c3 = st.columns(3)
        c1.metric('Top-1 accuracy (HMM)', f"{scores['hmm_top1']:.1%}")
        c2.metric('Top-1 classifier only', f"{scores['classifier_top1']:.1%}")
        c3.metric('Route continuity', f"{scores['hmm_continuity']:.1%}")
        c4, c5, c6 = st.columns(3)
        c4.metric('Route recall', f"{scores['hmm_route_recall']:.1%}")
        c5.metric('Route precision', f"{scores['hmm_route_precision']:.1%}")
        c6.metric('Processing time', f"{result['runtime_s']} s")
    else:
        st.info('No reference route for this trip, so no accuracy can be computed.')

    # map: raw GPS observations and the predicted route
    gps = result['gps']
    m = folium.Map(tiles='OpenStreetMap', control_scale=True)
    m.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)], [max(p[0] for p in gps), max(p[1] for p in gps)]])
    gps_layer = folium.FeatureGroup(name='Raw GPS observations')
    folium.PolyLine(gps, color='#1E88E5', weight=1.5, opacity=0.6).add_to(gps_layer)
    for p in gps[::max(1, len(gps) // 300)]:
        folium.CircleMarker(p, radius=3, color='#1565C0', fill=True, fill_opacity=0.8).add_to(gps_layer)
    gps_layer.add_to(m)
    route_layer = folium.FeatureGroup(name='Predicted route (HMM)')
    folium.PolyLine(result['predicted_route'], color='#E53935', weight=4).add_to(route_layer)
    route_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st.iframe(m.get_root().render(), height=600)  # map HTML built here by folium, not user input
    st.caption(f"Trip {trip_id}, {result['num_points']} GPS points. "
               "Blue: raw GPS observations. Red: predicted route after HMM/Viterbi decoding.")
