import { useEffect, useState } from "react";
import "./App.css";

const models = [
  {
    name: "Machine Learning",
    short: "ML",
    description: "Exoplanet habitability classification",
    technology: "Scikit-learn",
    status: "In Development",
    color: "blue",
  },
  {
    name: "Deep Learning",
    short: "DL",
    description: "Telescope light-curve analysis",
    technology: "PyTorch",
    status: "In Development",
    color: "purple",
  },
  {
    name: "Quantum ML",
    short: "QML",
    description: "Observation scheduling optimization",
    technology: "PennyLane / Qiskit",
    status: "In Development",
    color: "cyan",
  },
  {
    name: "LLM Benchmark",
    short: "LLM",
    description: "Benchmark against ML, DL and QML",
    technology: "Groq · GPT-OSS-20B",
    status: "Integrated",
    color: "orange",
  },
];

function App() {
  const [apiStatus, setApiStatus] = useState("Checking...");
  const [planet, setPlanet] = useState({
  planet_radius: 1.0,
  equilibrium_temperature: 280.0,
  insolation_flux: 1.0,
  star_temperature: 5778.0,
  star_mass: 1.0,
  star_radius: 1.0,
});

const [prediction, setPrediction] = useState(null);
const [loading, setLoading] = useState(false);
const [predictionError, setPredictionError] = useState("");
const [xgbPrediction, setXgbPrediction] = useState(null);
const [fluxInput, setFluxInput] = useState(
  Array(201).fill(1).join(", ")
);
const [dlPrediction, setDlPrediction] = useState(null);
const [dlLoading, setDlLoading] = useState(false);
const [dlError, setDlError] = useState("");
const [lstmPrediction, setLstmPrediction] = useState(null);
const [qmlTargets, setQmlTargets] = useState([
  { name: "Kepler-22b", duration: 2, priority: 90 },
  { name: "TOI-700d", duration: 1, priority: 80 },
  { name: "TRAPPIST-1e", duration: 2, priority: 70 },
  { name: "Proxima-b", duration: 1, priority: 60 },
]);

const [availableTime, setAvailableTime] = useState(3);
const [qmlSteps, setQmlSteps] = useState(100);
const [qmlResult, setQmlResult] = useState(null);
const [qmlLoading, setQmlLoading] = useState(false);
const [qmlError, setQmlError] = useState("");
// LLM Benchmark state
const [llmTargets, setLlmTargets] = useState([
  { name: "Kepler-22b", duration: 2, priority: 90 },
  { name: "TOI-700d", duration: 1, priority: 80 },
  { name: "TRAPPIST-1e", duration: 2, priority: 70 },
  { name: "Proxima-b", duration: 1, priority: 60 },
]);

const [llmResult, setLlmResult] = useState(null);
const [llmLoading, setLlmLoading] = useState(false);
const [llmError, setLlmError] = useState("");
const handlePredict = async (event) => {
  event.preventDefault();
  setLoading(true);
  setPrediction(null);
  setXgbPrediction(null);
  setPredictionError("");

  try {
    const payload = Object.fromEntries(
      Object.entries(planet).map(([key, value]) => [key, Number(value)])
    );

    const [rfResponse, xgbResponse] = await Promise.all([
      fetch("http://127.0.0.1:8000/planet/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
      fetch("http://127.0.0.1:8000/planet/predict/xgboost", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    ]);

    if (!rfResponse.ok || !xgbResponse.ok) {
      throw new Error("One or more planet prediction requests failed.");
    }

    const [rfData, xgbData] = await Promise.all([
      rfResponse.json(),
      xgbResponse.json(),
    ]);

    setPrediction(rfData);
    setXgbPrediction(xgbData);
  } catch (error) {
    setPredictionError(error.message);
  } finally {
    setLoading(false);
  }
};

const handleDLPredict = async (event) => {
  event.preventDefault();
  setDlLoading(true);
  setDlPrediction(null);
  setLstmPrediction(null);
  setDlError("");

  try {
    const flux = fluxInput
      .split(",")
      .map((value) => Number(value.trim()));

    if (flux.length !== 201 || flux.some((value) => !Number.isFinite(value))) {
      throw new Error("Please enter exactly 201 valid numbers.");
    }

    const payload = JSON.stringify({ flux });
    const options = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    };

    const [cnnResponse, lstmResponse] = await Promise.all([
      fetch("http://127.0.0.1:8000/transit/predict", options),
      fetch("http://127.0.0.1:8000/transit/predict/lstm", options),
    ]);

    if (!cnnResponse.ok || !lstmResponse.ok) {
      throw new Error("One or more transit prediction requests failed.");
    }

    const [cnnData, lstmData] = await Promise.all([
      cnnResponse.json(),
      lstmResponse.json(),
    ]);

    setDlPrediction(cnnData);
    setLstmPrediction(lstmData);
  } catch (error) {
    setDlError(error.message);
  } finally {
    setDlLoading(false);
  }
};

const handleQMLSchedule = async (event) => {
  event.preventDefault();

  setQmlLoading(true);
  setQmlResult(null);
  setQmlError("");

  try {
    const response = await fetch(
      "http://127.0.0.1:8000/schedule/compare",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          targets: qmlTargets.map((target) => ({
            ...target,
            duration: Number(target.duration),
            priority: Number(target.priority),
          })),
          available_time: Number(availableTime),
          steps: Number(qmlSteps),
        }),
      }
    );

    if (!response.ok) {
      throw new Error("QML scheduling request failed.");
    }

    const data = await response.json();
    setQmlResult(data);
  } catch (error) {
    setQmlError(error.message);
  } finally {
    setQmlLoading(false);
  }
};
   const handleLLMSchedule = async (event) => {
  event.preventDefault();

  setLlmLoading(true);
  setLlmResult(null);
  setLlmError("");

  try {
    const response = await fetch(
      "http://127.0.0.1:8000/llm/schedule",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          targets: llmTargets.map((target) => ({
            ...target,
            duration: Number(target.duration),
            priority: Number(target.priority),
          })),
          available_time: Number(availableTime),
        }),
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "LLM scheduling failed.");
    }

    const data = await response.json();
    setLlmResult(data);
  } catch (error) {
    setLlmError(error.message);
  } finally {
    setLlmLoading(false);
  }
};

useEffect(() => {
  fetch("http://127.0.0.1:8000/health")
    .then((response) => {
      if (!response.ok) {
        throw new Error("Backend unavailable");
      }
      return response.json();
    })
    .then((data) => {
      setApiStatus(data.status === "healthy" ? "Connected" : "Unavailable");
    })
    .catch(() => {
      setApiStatus("Disconnected");
    });
}, []);
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-icon">✦</span>
          <div>
            <h2>NEW WORLDS</h2>
            <small>EXOPLANET INTELLIGENCE</small>
          </div>
        </div>

        <p className="nav-label">WORKSPACE</p>
        <nav>
          <a className="active" href="#overview">◈ Overview</a>
          <a href="#models">◉ AI Models</a>
          <a href="#benchmark">▤ Benchmarking</a>
          <a href="#data">◫ Datasets</a>
          <a href="#about">ⓘ About Project</a>
        </nav>

        <div className="sidebar-bottom">
          <span className="live-dot" />
          System initialized
          <small>New Worlds v1.0.0</small>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <span>EXOPLANET RESEARCH PLATFORM</span>
          <span className="api-status">
          <span className="live-dot" />
          API {apiStatus}
          </span>
        </header>

        <section className="hero" id="overview">
          <p className="eyebrow">INTELLIGENT SPACE EXPLORATION</p>
          <h1>Discover <span>New Worlds.</span></h1>
          <p>
            An intelligent platform combining ML, DL, Quantum ML and
            Large Language Models to analyze exoplanets and optimize
            astronomical observations.
          </p>
          <div className="hero-tags">
            <span>Machine Learning</span>
            <span>Deep Learning</span>
            <span>Quantum Computing</span>
            <span>LLM Benchmarking</span>
          </div>
        </section>

        <section className="stats">
          <div className="stat-card">
            <small>AI COMPONENTS</small>
            <h2>04</h2>
            <span>ML · DL · QML · LLM</span>
          </div>
          <div className="stat-card">
            <small>DATASETS</small>
            <h2>--</h2>
            <span>Not integrated yet</span>
          </div>
          <div className="stat-card">
            <small>EXPERIMENTS</small>
            <h2>00</h2>
            <span>Awaiting first model run</span>
          </div>
          <div className="stat-card">
            <small>BENCHMARK STATUS</small>
            <h2 className="pending">Pending</h2>
            <span>Evaluation not started</span>
          </div>
        </section>

        <section className="models-section" id="models">
          <div className="section-heading">
            <div>
              <p className="eyebrow">INTELLIGENCE LAYER</p>
              <h2>AI Model Ecosystem</h2>
            </div>
            <span className="section-count">04 COMPONENTS</span>
          </div>

          <div className="model-grid">
            {models.map((model) => (
              <article className="model-card" key={model.short}>
                <div className={`model-icon ${model.color}`}>
                  {model.short}
                </div>
                <span className="model-status">{model.status}</span>
                <h3>{model.name}</h3>
                <p>{model.description}</p>
                <div className="technology">{model.technology}</div>
              </article>
            ))}
          </div>
        </section>

<section className="prediction-section" id="qml-scheduler">
  <div className="section-heading">
    <div>
      <p className="eyebrow">QUANTUM ML · PENNYLANE QAOA</p>
      <h2>Observation Scheduler</h2>
      <p>Optimize telescope observation targets within a limited time.</p>
    </div>
  </div>

  <form
    className="prediction-form"
    onSubmit={handleQMLSchedule}
  >
    {qmlTargets.map((target, index) => (
      <div className="qml-target" key={index}>
        <h3>Target {index + 1}</h3>

        <label>
          Planet Name
          <input
            value={target.name}
            onChange={(e) => {
              const updated = [...qmlTargets];
              updated[index] = {
                ...target,
                name: e.target.value,
              };
              setQmlTargets(updated);
            }}
            required
          />
        </label>

        <label>
          Observation Duration (hours)
          <input
            type="number"
            min="0.01"
            step="any"
            value={target.duration}
            onChange={(e) => {
              const updated = [...qmlTargets];
              updated[index] = {
                ...target,
                duration: e.target.value,
              };
              setQmlTargets(updated);
            }}
            required
          />
        </label>

        <label>
          Priority
          <input
            type="number"
            min="0"
            step="any"
            value={target.priority}
            onChange={(e) => {
              const updated = [...qmlTargets];
              updated[index] = {
                ...target,
                priority: e.target.value,
              };
              setQmlTargets(updated);
            }}
            required
          />
        </label>
      </div>
    ))}

    <label>
      Available Telescope Time (hours)
      <input
        type="number"
        min="0"
        step="any"
        value={availableTime}
        onChange={(e) => setAvailableTime(e.target.value)}
        required
      />
    </label>

    <label>
      QAOA Optimization Steps
      <input
        type="number"
        min="1"
        max="500"
        value={qmlSteps}
        onChange={(e) => setQmlSteps(e.target.value)}
        required
      />
    </label>

    <button type="submit" disabled={qmlLoading}>
      {qmlLoading ? "Optimizing Schedule..." : "Run Quantum Scheduler"}
    </button>
  </form>

  {qmlError && (
    <p className="prediction-error">{qmlError}</p>
  )}

  {qmlResult && (
    <div className="prediction-result">
      <h3>Observation Scheduling Results</h3>

      <h4>Classical Scheduler</h4>
      <p>Total Priority: {qmlResult.classical_result.total_priority}</p>
      <p>
        Observation Time:{" "}
        {qmlResult.classical_result.total_observation_time} hours
      </p>

      <h4>Quantum Scheduler — QAOA</h4>
      <p>Total Priority: {qmlResult.quantum_result.total_priority}</p>
      <p>
        Observation Time:{" "}
        {qmlResult.quantum_result.total_observation_time} hours
      </p>
      <p>
        Matches Classical:{" "}
        {qmlResult.comparison.qaoa_matches_classical ? "Yes" : "No"}
      </p>

      <h4>Quantum Scheduler — VQE</h4>
      <p>Total Priority: {qmlResult.vqe_result.total_priority}</p>
      <p>
        Observation Time:{" "}
        {qmlResult.vqe_result.total_observation_time} hours
      </p>
      <p>
        Matches Classical:{" "}
        {qmlResult.comparison.vqe_matches_classical ? "Yes" : "No"}
      </p>

      <h4>VQE Selected Targets</h4>
      {qmlResult.vqe_result.selected_targets.map((target) => (
        <p key={target.name}>
          {target.name} — {target.duration} hours · Priority {target.priority}
        </p>
      ))}

      <p>Quantum advantage claimed: No</p>
    </div>
  )}
</section>

        <section className="prediction-section" id="prediction">
  <div className="section-heading">
    <div>
      <p className="eyebrow">MACHINE LEARNING · RANDOM FOREST</p>
      <h2>Exoplanet Habitability Predictor</h2>
    </div>
  </div>

  <form onSubmit={handlePredict} className="prediction-form">
    {[
      ["planet_radius", "Planet Radius (Earth radii)"],
      ["equilibrium_temperature", "Equilibrium Temperature (K)"],
      ["insolation_flux", "Insolation Flux"],
      ["star_temperature", "Star Temperature (K)"],
      ["star_mass", "Star Mass (Solar masses)"],
      ["star_radius", "Star Radius (Solar radii)"],
    ].map(([key, label]) => (
      <label key={key}>
        {label}
        <input
          type="number"
          step="any"
          min="0.000001"
          required
          value={planet[key]}
          onChange={(e) =>
            setPlanet({ ...planet, [key]: e.target.value })
          }
        />
      </label>
    ))}

    <button type="submit" disabled={loading}>
      {loading ? "Analyzing Planet..." : "Predict Habitability"}
    </button>
  </form>

  {prediction && (
    <div className="prediction-result">
      <h3>{prediction.classification}</h3>
      <p>Model: {prediction.model}</p>
      <p>{prediction.disclaimer}</p>
    </div>
  )}

  {xgbPrediction && (
    <div className="prediction-result">
      <h3>XGBoost Result</h3>
      <p>Classification: {xgbPrediction.classification}</p>
      <p>Model: {xgbPrediction.model}</p>
      <p>{xgbPrediction.disclaimer}</p>
    </div>
  )}
  {predictionError && (
    <p className="prediction-error">{predictionError}</p>
  )}
</section>

<section className="prediction-section" id="dl-prediction">
  <div className="section-heading">
    <div>
      <p className="eyebrow">DEEP LEARNING · PYTORCH CNN</p>
      <h2>Transit Signal Classifier</h2>
    </div>
  </div>

  <form onSubmit={handleDLPredict} className="prediction-form">
    <label>
      Telescope Flux Measurements (exactly 201 comma-separated values)
      <textarea
        rows="6"
        value={fluxInput}
        onChange={(e) => setFluxInput(e.target.value)}
        required
      />
    </label>

    <button type="submit" disabled={dlLoading}>
      {dlLoading ? "Analyzing Signal..." : "Classify Transit Signal"}
    </button>
  </form>

  {dlPrediction && (
    <div className="prediction-result">
      <h3>{dlPrediction.classification}</h3>
      <p>Prediction: {dlPrediction.prediction}</p>
      <p>Confidence: {(dlPrediction.confidence * 100).toFixed(2)}%</p>
      <p>Test with real telescope observations for meaningful evaluation.</p>
    </div>
  )}

  {lstmPrediction && (
    <div className="prediction-result">
      <h3>TransitLSTM Result</h3>
      <p>Classification: {lstmPrediction.classification}</p>
      <p>Prediction: {lstmPrediction.prediction}</p>
      <p>Confidence: {(lstmPrediction.confidence * 100).toFixed(2)}%</p>
    </div>
  )}
  {dlError && <p className="prediction-error">{dlError}</p>}
</section>
<section className="prediction-section" id="llm-scheduler">
  <div className="section-heading">
    <div>
      <p className="eyebrow">LARGE LANGUAGE MODEL · GROQ</p>
      <h2>AI Observation Scheduler</h2>
      <p>Use GPT-OSS-20B to select observation targets within telescope time limits.</p>
    </div>
  </div>

  <form className="prediction-form" onSubmit={handleLLMSchedule}>
    {llmTargets.map((target, index) => (
      <div className="qml-target" key={index}>
        <h3>Target {index + 1}</h3>

        <label>
          Planet Name
          <input
            value={target.name}
            onChange={(e) => {
              const updated = [...llmTargets];
              updated[index] = { ...target, name: e.target.value };
              setLlmTargets(updated);
            }}
            required
          />
        </label>

        <label>
          Observation Duration (hours)
          <input
            type="number"
            min="0.01"
            step="any"
            value={target.duration}
            onChange={(e) => {
              const updated = [...llmTargets];
              updated[index] = { ...target, duration: e.target.value };
              setLlmTargets(updated);
            }}
            required
          />
        </label>

        <label>
          Priority
          <input
            type="number"
            min="0"
            step="any"
            value={target.priority}
            onChange={(e) => {
              const updated = [...llmTargets];
              updated[index] = { ...target, priority: e.target.value };
              setLlmTargets(updated);
            }}
            required
          />
        </label>
      </div>
    ))}

    <label>
      Available Telescope Time (hours)
      <input
        type="number"
        min="0"
        step="any"
        value={availableTime}
        onChange={(e) => setAvailableTime(e.target.value)}
        required
      />
    </label>

    <button type="submit" disabled={llmLoading}>
      {llmLoading ? "Groq AI Scheduling..." : "Run LLM Scheduler"}
    </button>
  </form>

  {llmError && (
    <p className="prediction-error">{llmError}</p>
  )}

  {llmResult && (
    <div className="prediction-result">
      <h3>Groq Scheduling Results</h3>
      <p>Model: {llmResult.model}</p>
      <p>Selected Targets:</p>

      {llmResult.selected_targets.map((name) => (
        <p key={name}>✦ {name}</p>
      ))}

      <p>Total Observation Time: {llmResult.total_duration} hours</p>
      <p>Total Priority: {llmResult.total_priority}</p>
      <p>Available Time: {llmResult.available_time} hours</p>
      <p>Reasoning: {llmResult.reasoning}</p>
    </div>
  )}
</section>
<section className="benchmark-section" id="benchmark">
          <div>
            <p className="eyebrow">PERFORMANCE EVALUATION</p>
            <h2>Model Benchmarking</h2>
            <p>
              Compare our ML, DL and QML approaches against an LLM
              benchmark using task-specific evaluation metrics.
            </p>
          </div>
          <div className="benchmark-placeholder">
            <span>◈</span>
            <h3>Benchmarking not started</h3>
            <p>Results will appear here after model implementation.</p>
          </div>
        </section>

        <footer id="about">
          NEW WORLDS © 2026 · EXOPLANET ANALYSIS PLATFORM
        </footer>
      </main>
    </div>
  );
}

export default App;