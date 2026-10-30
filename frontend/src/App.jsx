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
    technology: "Gemini / OpenAI",
    status: "In Development",
    color: "orange",
  },
];

function App() {
  const [apiStatus, setApiStatus] = useState("Checking...");

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
