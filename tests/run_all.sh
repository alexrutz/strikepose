#!/usr/bin/env bash
# Runs every check. GUI tests need a display; on a headless box install xvfb and
# they will be wrapped automatically.
set -u
cd "$(dirname "$0")/.."
RUN=""
if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null; then RUN="xvfb-run -a"; fi
fail=0
run() {
  printf '%-34s' "$1"
  out=$($2 2>&1)
  if printf '%s' "$out" | grep -qE '^FAIL|FAILURES'; then
    echo "FAILED"; printf '%s\n' "$out" | grep -E '^FAIL' | head -5; fail=1
  elif printf '%s' "$out" | grep -qE 'ALL PASS|^OK|DONE'; then echo "ok"
  else echo "FAILED"; printf '%s\n' "$out" | tail -20; fail=1; fi
}
run "core maths"            "python3 tests/test_core.py"
run "rigged mesh selftest"  "python3 mesh_backend.py --selftest"
run "smpl-x selftest"       "python3 smplx_backend.py --selftest"
run "pose agent selftest"   "python3 pose_agent.py --selftest"
run "pose agent wire"       "python3 tests/test_pose_agent.py"
for t in tests/test_editor_basics.py tests/test_depth_export.py \
         tests/test_body_presets.py tests/test_legacy_scenes.py \
         tests/test_smplx_fallback.py tests/test_multi_person.py \
         tests/test_rigged_mesh.py tests/test_depth_sources.py \
         tests/test_editing_modes.py tests/test_panel_layout.py \
         tests/test_anchors_hinge.py tests/test_prompt_ui.py; do
  run "$(basename "$t" .py)" "$RUN python3 $t"
done
if command -v node >/dev/null; then
  run "web core"    "node web/test.mjs"
  run "web logic"   "node web/headless.mjs"
fi
exit $fail
