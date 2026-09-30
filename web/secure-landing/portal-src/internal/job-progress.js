// A zero progress value is also the initial value for jobs that do not emit
// granular progress. Transport heartbeats must never manufacture a percentage.
export function jobProgressSnapshot(job) {
  if (!job) return { value: 0, label: "No run selected", waiting: false };
  if (job.state === "succeeded") return { value: 100, label: "100%", waiting: false };

  const reported = Number(job.progress);
  const value = Number.isFinite(reported) ? Math.max(0, Math.min(100, reported)) : 0;
  const waiting = value === 0 && (job.state === "queued" || job.state === "running");
  let label = `${value}%`;
  if (value === 0) {
    switch (job.state) {
      case "queued": label = "Queued"; break;
      case "running": label = "In progress"; break;
      case "failed": label = "Failed"; break;
      case "canceled": label = "Canceled"; break;
      case "partial": label = "Partial result"; break;
      case "offline": label = "Offline"; break;
      default: label = "Not reported";
    }
  }
  return {
    value,
    label,
    waiting,
  };
}

export function renderJobProgress(element, snapshot) {
  if (!element) return;
  element.max = 100;
  if (snapshot.waiting) element.removeAttribute("value");
  else element.value = snapshot.value;
  element.dataset.progressState = snapshot.waiting ? "waiting" : snapshot.value > 0 ? "measured" : "unavailable";
  element.setAttribute("aria-valuetext", snapshot.label);
}
