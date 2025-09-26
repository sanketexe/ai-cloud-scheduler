const API_BASE_URL = "http://localhost:8000";

// Dashboard
function loadSampleData() {
  fetch(`${API_BASE_URL}/api/workloads/sample`)
    .then(res => res.json())
    .then(data => {
      document.getElementById("workloadCount").innerText = data.workloads.length;
      alert("Sample workloads loaded!");
    });
}

function runQuickSimulation() {
  fetch(`${API_BASE_URL}/api/simulation/start`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({schedulers: ["random", "lowest_cost", "round_robin"]})
  }).then(res => res.json())
    .then(() => document.getElementById("lastSim").innerText = "✅ Completed");
}

// Config
function loadProviders() {
  fetch(`${API_BASE_URL}/api/providers/default`).then(r => r.json()).then(console.log);
}

function loadVMs() {
  fetch(`${API_BASE_URL}/api/vms/default`).then(r => r.json()).then(console.log);
}

function uploadWorkloads() {
  alert("Upload workloads feature coming soon!");
}

// Simulation
function startSimulation() {
  const schedulers = [...document.querySelectorAll("input[type=checkbox]:checked")]
    .map(cb => cb.value);

  fetch(`${API_BASE_URL}/api/simulation/start`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({schedulers})
  }).then(r => r.json()).then(data => {
    alert("Simulation completed!");
    window.location.href = "results.html";
  });
}

// Results
function loadResults() {
  fetch(`${API_BASE_URL}/api/simulation/results`)
    .then(r => r.json())
    .then(data => {
      const table = document.getElementById("resultsTable");
      data.forEach(res => {
        let row = `<tr>
          <td>${res.scheduler}</td>
          <td>${res.cost}</td>
          <td>${res.latency}</td>
          <td>${res.sla_violations}</td>
        </tr>`;
        table.innerHTML += row;
      });
    });
}

// ML Predictions
function runMLPrediction() {
  fetch(`${API_BASE_URL}/api/ml/predict`)
    .then(r => r.json())
    .then(data => {
      document.getElementById("predictionOutput").innerText =
        "Best Scheduler: " + data.best_scheduler;
    });
}

// Auto-run results loader
if (window.location.pathname.includes("results.html")) {
  loadResults();
}
