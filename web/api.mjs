// The engine, over HTTP.
//
// Everything that decides what comes *out* of the editor lives in the Python:
// the pose catalogue, the objects, the clothing, the prompt, and above all the
// export. The phone keeps the projection and the drag, because those have to
// answer in a frame, and asks for the rest. The alternative - a second
// implementation of the same geometry in JavaScript - is what this repo
// already had, and it had drifted by a third of the child's torso before
// anything compared the two.
//
// If nothing answers, `online` goes false and the app falls back to what it
// can do alone. That is worth keeping: the page still poses a figure on a
// train with no laptop, it just cannot export the real depth map.

export const state = {online: false, vocabulary: null, reason: ""};

async function call(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).error || detail; } catch (e) {}
    throw new Error(detail);
  }
  return response.json();
}

export async function connect() {
  try {
    const response = await fetch("api/vocabulary", {cache: "no-store"});
    if (!response.ok) throw new Error("HTTP " + response.status);
    state.vocabulary = await response.json();
    state.online = true;
  } catch (problem) {
    state.online = false;
    state.reason = String(problem.message || problem);
  }
  return state;
}

// A scene the server works out: a named pose, or a plan from a prompt.
export const scene = plan => call("api/scene", {plan});

// A scene the phone has already posed. `people` carries the keypoints the
// drag produced, so the render is of exactly what is on screen.
export const render = payload => call("api/render", payload);

export const prompt = (text, host) => call("api/prompt", {prompt: text, host});
