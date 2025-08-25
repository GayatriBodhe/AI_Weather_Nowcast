import React, { useMemo, useState } from "react";
import { Layers, MapPin, Wind, Eye, Droplets, Gauge, Map, Cloud, Sun, CloudRain, CloudLightning } from "lucide-react";
import { MapContainer, TileLayer, FeatureGroup, Rectangle } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// If you don't have shadcn/ui, use light wrappers
const Button = ({ className = "", children, ...rest }) => (
  <button
    className={`px-3 py-2 rounded-2xl shadow-sm border bg-white hover:bg-gray-50 active:scale-[.99] transition ${className}`}
    {...rest}
  >
    {children}
  </button>
);
const Card = ({ className = "", children }) => (
  <div className={`rounded-2xl border shadow-sm bg-white ${className}`}>{children}</div>
);
const CardHeader = ({ children, className = "" }) => (
  <div className={`p-4 border-b ${className}`}>{children}</div>
);
const CardTitle = ({ children, className = "" }) => (
  <h2 className={`text-xl font-semibold ${className}`}>{children}</h2>
);
const CardContent = ({ children, className = "" }) => (
  <div className={`p-4 ${className}`}>{children}</div>
);

// ---- Helpers ----
function classNames(...xs) {
  return xs.filter(Boolean).join(" ");
}

// ---- Main Component ----
export default function NowcastUI() {
  const [location] = useState("Mumbai, India");

  // Demo weather data
  const weatherData = {
    temperature: "28°C",
    wind: "15 km/h",
    visibility: "8 km",
    humidity: "70%",
    pressure: "1012 hPa",
  };

  const forecast = [
    { day: "Today", date: "Aug 24", tempMax: "28°C", tempMin: "18°C", condition: "Cloudy", rain: "48%" },
    { day: "Tomorrow", date: "Aug 25", tempMax: "28°C", tempMin: "18°C", condition: "Cloudy", rain: "64%" },
    { day: "Wed", date: "Aug 26", tempMax: "29°C", tempMin: "19°C", condition: "Sunny", rain: "51%" },
    { day: "Thu", date: "Aug 27", tempMax: "30°C", tempMin: "20°C", condition: "Sunny", rain: "18%" },
    { day: "Fri", date: "Aug 28", tempMax: "21°C", tempMin: "11°C", condition: "Cloudy", rain: "82%" },
    { day: "Sat", date: "Aug 29", tempMax: "24°C", tempMin: "14°C", condition: "Light Rain", rain: "13%" },
    { day: "Sun", date: "Aug 30", tempMax: "23°C", tempMin: "13°C", condition: "Sunny", rain: "64%" },
  ];

  const getIcon = (condition) => {
    switch (condition) {
      case "Sunny":
        return <Sun className="text-yellow-500 w-6 h-6" />;
      case "Cloudy":
        return <Cloud className="text-gray-500 w-6 h-6" />;
      case "Light Rain":
      case "Rainy":
        return <CloudRain className="text-blue-500 w-6 h-6" />;
      case "Storm":
        return <CloudLightning className="text-purple-600 w-6 h-6" />;
      default:
        return <Cloud className="text-gray-500 w-6 h-6" />;
    }
  };

  // Map state
  const [center, setCenter] = useState([19.0760, 72.8777]); // India center
  const [zoom, setZoom] = useState(10);

  // Geo bounds for overlays (Leaflet LatLngBounds)
  const defaultBounds = useMemo(
    () => new L.LatLngBounds([18.85, 72.6], [19.33, 73.3]), 
    []
  );
  const [bounds] = useState(defaultBounds);

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-50 to-blue-50 text-slate-800">
      {/* Header */}
      <div className="max-w-7xl mx-auto px-4 py-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-2xl bg-white shadow-sm border">
            <Map className="w-6 h-6" />
          </div>
          <div>
            <div className="text-2xl font-bold">RainCast Nowcasting</div>
            <div className="text-sm text-gray-600">ConvLSTM / U-Net demo UI for radar/satellite sequences</div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 pb-10 grid grid-cols-12 gap-4">
        {/* Map & overlays */}
        <div className={classNames("col-span-12","lg:col-span-9") }>
          <Card className="shadow-xl rounded-2xl">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center gap-2"><Layers className="w-5 h-5"/> Overlay</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {/* Map */}
              <div className="w-full h-[60vh] rounded-xl overflow-hidden">
                <MapContainer
                  center={center}
                  zoom={zoom}
                  style={{ width: "100%", height: "100%" }}
                  className="rounded-xl"
                  scrollWheelZoom
                  whenReady={(m) => {
                    const map = m.target;
                    map.on("moveend", () => setCenter([map.getCenter().lat, map.getCenter().lng]));
                    map.on("zoomend", () => setZoom(map.getZoom()));
                  }}
                >
                  <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution="&copy; OpenStreetMap contributors"
                  />

                  {/* Show rectangle of bounds */}
                  <FeatureGroup>
                    <Rectangle
                      bounds={bounds}
                      pathOptions={{ color: "#2563eb", weight: 1 }}
                    />
                  </FeatureGroup>
                </MapContainer>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Side panel */}
        <div className={classNames("col-span-12", "lg:col-span-3") }>
          <div className="grid gap-4">
            <Card className="shadow-xl">
              <CardHeader>
                <CardTitle>Region Bounds</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between"><span>South</span><span className="tabular-nums">{bounds.getSouth().toFixed(3)}</span></div>
                  <div className="flex justify-between"><span>West</span><span className="tabular-nums">{bounds.getWest().toFixed(3)}</span></div>
                  <div className="flex justify-between"><span>North</span><span className="tabular-nums">{bounds.getNorth().toFixed(3)}</span></div>
                  <div className="flex justify-between"><span>East</span><span className="tabular-nums">{bounds.getEast().toFixed(3)}</span></div>
                </div>
              </CardContent>
            </Card>
            <Card className="shadow-xl rounded-2xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MapPin className="h-5 w-5 text-blue-500" />
                  {location}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {/* Temperature Row */}
                <div className="text-4xl font-bold text-center text-blue-600 mb-4">
                  {weatherData.temperature}
                </div>

                {/* Weather Details Row */}
                <div className="grid grid-cols-2 gap-4 text-center text-sm">
                  <div className="flex flex-col items-center">
                    <Wind className="h-5 w-5 text-gray-500 mb-1" />
                    <p className="font-medium">{weatherData.wind}</p>
                    <p className="text-gray-500">Wind</p>
                  </div>
                  <div className="flex flex-col items-center">
                    <Eye className="h-5 w-5 text-gray-500 mb-1" />
                    <p className="font-medium">{weatherData.visibility}</p>
                    <p className="text-gray-500">Visibility</p>
                  </div>
                  <div className="flex flex-col items-center">
                    <Droplets className="h-5 w-5 text-gray-500 mb-1" />
                    <p className="font-medium">{weatherData.humidity}</p>
                    <p className="text-gray-500">Humidity</p>
                  </div>
                  <div className="flex flex-col items-center">
                    <Gauge className="h-5 w-5 text-gray-500 mb-1" />
                    <p className="font-medium">{weatherData.pressure}</p>
                    <p className="text-gray-500">Pressure</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* <div className="max-w-7xl mx-auto px-4 pb-10 grid grid-cols-12 gap-4">
        <Card className="shadow-xl rounded-2xl lg:col-span-12 p-4">
          <CardHeader>
            <CardTitle className="text-xl font-semibold">Model Predictions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex justify-center items-center">
              <img
                src="/public/Predictions.png"
                alt="predictions Graph"
                className="rounded-xl shadow-md w-full h-auto object-contain"
              />
            </div>
          </CardContent>
        </Card>
      </div> */}
      <div className="max-w-7xl mx-auto px-4 pb-10 grid grid-cols-12 gap-4">
        <Card className="shadow-xl rounded-2xl lg:col-span-12 p-4">
          <CardHeader>
            <CardTitle className="text-xl font-semibold">Model Predictions</CardTitle>
          </CardHeader>
          <CardContent>
            {/* Input sequence full width */}
            <div className="flex flex-col mb-6">
              <h3 className="font-medium mb-5 text-xl">Input Sequence</h3>
              <img
                src="/input_seq.png"
                alt="Input Sequence"
                className="rounded-xl w-full h-auto object-contain"
              />
            </div>

            {/* Actual and Predicted side by side */}
            <div className="grid grid-cols-2 gap-4 items-center">
              <div className="flex flex-col items-center">
              <img
                src="/actual.png"
                alt="Actual Frame"
                className="rounded-xl w-fit h-auto object-contain"
              />
            </div>
              <div className="flex flex-col items-center">

                <img
                  src="/predicted.png"
                  alt="Predicted Frame"
                  className="rounded-xl w-fit h-auto object-contain"
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="max-w-7xl mx-auto px-4 pb-10 grid grid-cols-12 gap-4">
        <Card className="shadow-xl rounded-2xl lg:col-span-7 p-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-xl font-semibold flex items-center gap-2">
              7-Day Forecast
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="divide-y divide-gray-200">
              {forecast.map((day, idx) => (
                <div
                  key={idx}
                  className="flex justify-between items-center py-3 hover:bg-gray-50 rounded-lg transition"
                >
                  {/* Left side - Day & Date */}
                  <div className="w-1/4">
                    <p className="font-semibold">{day.day}</p>
                    <p className="text-xs text-gray-500">{day.date}</p>
                  </div>

                  {/* Icon + Condition */}
                  <div className="flex items-center gap-2 w-1/3">
                    {getIcon(day.condition)}
                    <div className="text-sm">
                      <p className="font-medium">{day.condition}</p>
                      <p className="text-xs text-blue-600">{day.rain} rain</p>
                    </div>
                  </div>

                  {/* Temperature */}
                  <div className="text-right w-1/4">
                    <p className="text-lg font-semibold">{day.tempMax}</p>
                    <p className="text-sm text-gray-500">{day.tempMin}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="shadow-xl rounded-2xl lg:col-span-5 p-4">
          <CardHeader>
            <CardTitle className="text-xl font-semibold">Accuracy level of the model in training</CardTitle>
          </CardHeader>
          <CardContent>
            <div>
              <div className="flex justify-center items-center">
                <img
                  src="/public/Accuracy.jpeg"
                  alt="Model Accuracy Graph"
                  className="rounded-xl shadow-md w-full h-auto object-contain"
                />
              </div>
              <br />
              <p className="text-lg">Training accuracy curve showing how the model’s performance improved over epochs</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <footer className="max-w-7xl mx-auto px-4 pb-6 text-xs text-gray-500">
        @{new Date().getFullYear()} AI Nowcast. All right reserved.
      </footer>
    </div>
  );
}
