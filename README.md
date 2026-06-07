# CrisisRoute v4

CrisisRoute is a real-time emergency routing dashboard developed for the **CSE 208 Algorithms Semester Project**.

The project simulates how a rescue unit can find a route in a damaged city or campus area. It combines a real road graph, map-based start/destination selection, optional aerial image analysis with Gemini Vision, and classical graph algorithms.

> Gemini helps the system understand visual input. The actual routing decision is still made by Dijkstra's Algorithm and Priority Queue logic.

---

## Main Features

- Real road graph loading with **OpenStreetMap + OSMnx**
- Built-in offline demo map
- Start and destination selection directly from the map
- Dijkstra-based route calculation
- Emergency target selection with Priority Queue
- Road closure support
- Live route recalculation after graph updates
- Aerial/drone image upload
- Gemini Vision suggestions for blocked roads and risk areas
- User confirmation before AI suggestions are applied
- Real map background with graph overlay
- Clean dashboard interface with route legend

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web dashboard |
| Folium / streamlit-folium | Interactive real map and click selection |
| Plotly | Demo graph visualization |
| OSMnx | Loading real OpenStreetMap road graphs |
| NetworkX | Graph support for OSMnx |
| Gemini API | Aerial image analysis |
| heapq | Priority Queue implementation |

---

## Algorithmic Logic

The city map is represented as a graph:

- Nodes represent road intersections or important locations.
- Edges represent roads between these points.
- Closed roads are removed from routing.
- Dijkstra's Algorithm finds the lowest-cost route.
- Priority Queue selects the most urgent reachable target in emergency mode.

In simple terms:

```text
Priority Queue decides where to go.
Dijkstra decides how to get there.
```

The Dijkstra implementation uses a heap-based structure, so the complexity is:

```text
O((V + E) log V)
```

---

## AI Integration

Gemini Vision is used to analyze uploaded aerial or drone-like images.

It can suggest:

- possible blocked roads
- risky areas
- useful places such as parks, open areas or water sources

The AI output is not applied automatically. The user first reviews the suggestions, then confirms which road closures should update the graph.

This keeps the AI as a support module, not the final decision-maker.

---

## Requirements

Python 3.10 or newer is recommended.

Install the required packages:

```bash
pip install -r requirements.txt
```

The main requirements are:

```txt
streamlit
plotly
networkx
osmnx
google-genai
pillow
folium
streamlit-folium
```

---

## Gemini API Key

Image analysis requires a Gemini API key.

Windows:

```bash
set GEMINI_API_KEY=your_key_here
```

macOS / Linux:

```bash
export GEMINI_API_KEY=your_key_here
```

You can also enter the API key inside the app.

Do not upload your API key to GitHub.

---

## How to Run

Open a terminal in the project folder and run:

```bash
streamlit run app.py
```

The app will open in your browser.

---

## Recommended Demo Flow

1. Start the app.
2. Load the demo map or load a real road graph.
3. Select the start point from the map.
4. Select the destination point from the map.
5. Calculate the route.
6. Upload an aerial or satellite-like image.
7. Analyze it with Gemini Vision.
8. Apply suggested road closures.
9. Watch the graph update and the route recalculate.

---

## Project Structure

```text
crisisroute/
│
├── app.py
├── requirements.txt
├── README.md
├── run_windows.bat
└── run_mac_linux.sh
```

---

## Limitations

- Gemini may not always detect exact road names correctly from aerial images.
- OpenStreetMap geocoding may fail for small campus buildings.
- The uploaded image is not fully geo-referenced yet.
- AI suggestions still require user confirmation.
- The system is a decision-support prototype, not a fully autonomous disaster management platform.

---

## Future Work

Future versions can include:

- real-time drone image feed
- automatic geo-referencing of uploaded images
- stronger vision segmentation models
- multi-vehicle emergency routing
- live GPS tracking
- automatic road damage detection
- TÜBİTAK 2209-style research extension

---

## AI Usage Transparency

AI tools were used as a development assistant and Gemini Vision is used as an optional image-analysis module.

However, the core algorithmic logic is based on classical algorithms:

- Dijkstra's Algorithm
- graph representation
- heap-based Priority Queue

The AI module suggests updates. The graph algorithms make the route decision.

---

## GitHub Repository

```text
[https://github.com/your-username/crisisroute](https://github.com/semih896/crisisroute)
```
