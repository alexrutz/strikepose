#!/usr/bin/env python3
"""Settings that outlive the checkout.

The file lives in the PARENT of the application directory, not inside it, so
that `rm -rf strikepose && git clone ...` keeps the one thing nobody wants to
type again: the address of a model server on another machine, and the key for
it. Anything inside the repo is by definition something a clean clone throws
away.

    ../strikepose-settings.json

`$STRIKEPOSE_SETTINGS` overrides the path outright, for anyone who would
rather keep it somewhere else or run two configurations side by side.

It is written 0600 because it holds an API key, and read defensively because
a settings file is the one file a user WILL hand-edit: a missing key, a
string where a number goes, or a truncated write after a power cut all have
to leave the editor starting normally with the default for that one field
rather than not starting at all.
"""

from __future__ import annotations

import json
import os
import tempfile

FILENAME = "strikepose-settings.json"

# What is worth keeping between runs, with the default for each. Anything not
# in here is not persisted - a pose belongs in a scene file, not in settings.
DEFAULTS = {
    "llm_host": "",
    "llm_model": "",
    "llm_backend": "auto",
    "llm_key": "",
    "llm_timeout": 600,
    "llm_passes": 2,
    "reasoning": "xhigh",
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "max_tokens": "",
    "seed": "",
    "export_w": 512,
    "export_h": 768,
    "preset": "",
    "window": "",
}

SECRET = ("llm_key",)


def app_dir():
    """Where the application lives."""
    return os.path.dirname(os.path.abspath(__file__))


def path():
    """The settings file, outside the checkout."""
    named = os.environ.get("STRIKEPOSE_SETTINGS")
    if named:
        return os.path.abspath(os.path.expanduser(named))
    return os.path.join(os.path.dirname(app_dir()), FILENAME)


def load(where=None):
    """Everything in DEFAULTS, overlaid with whatever the file has that fits.

    A value of the wrong type is dropped rather than coerced: a `top_k` of
    "twenty" should leave top_k at 20 and the rest of the file working, not
    raise somewhere three modules away when the sampler is built.
    """
    out = dict(DEFAULTS)
    try:
        with open(where or path(), "r", encoding="utf-8") as handle:
            stored = json.load(handle)
    except (OSError, ValueError):
        return out
    if not isinstance(stored, dict):
        return out
    for key, fallback in DEFAULTS.items():
        if key not in stored:
            continue
        value = stored[key]
        if isinstance(fallback, bool):
            if isinstance(value, bool):
                out[key] = value
        elif isinstance(fallback, int) and not isinstance(fallback, bool):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = int(value)
        elif isinstance(fallback, float):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = float(value)
        elif isinstance(fallback, str):
            if isinstance(value, (str, int, float)):
                out[key] = str(value)
    return out


def save(values, where=None):
    """Write the settings, atomically and privately.

    Atomic because the alternative is a half-written file when something goes
    wrong mid-save, and a half-written settings file is worse than none: it
    parses as far as the truncation and then does not. Written to a temporary
    file in the same directory and renamed, which is atomic on every platform
    that matters.

    Returns the path written, or None if it could not be - a read-only parent
    directory is a reason to carry on without saved settings, not a reason to
    refuse to run.
    """
    target = where or path()
    keep = {k: values[k] for k in DEFAULTS if k in values}
    try:
        folder = os.path.dirname(target) or "."
        os.makedirs(folder, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=folder, prefix=".settings-")
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(keep, out, indent=1, sort_keys=True)
            out.write("\n")
        os.replace(temporary, target)
        if any(keep.get(name) for name in SECRET):
            os.chmod(target, 0o600)
        return target
    except OSError:
        try:
            os.unlink(temporary)
        except (OSError, NameError, UnboundLocalError):
            pass
        return None


def describe(where=None):
    target = where or path()
    return "%s (%s)" % (target, "saved" if os.path.exists(target) else "not yet")


def _selftest():
    import shutil
    ok = True

    def check(label, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(("PASS " if cond else "FAIL ") + label + ("  " + extra if extra else ""))

    folder = tempfile.mkdtemp()
    target = os.path.join(folder, "s.json")

    check("a missing file gives the defaults", load(target) == DEFAULTS)

    values = dict(DEFAULTS, llm_host="http://10.0.0.7:8080", llm_key="sk-x",
                  temperature=0.85, top_k=40, reasoning="medium")
    save(values, target)
    back = load(target)
    check("what is saved comes back", back["llm_host"] == "http://10.0.0.7:8080"
          and back["temperature"] == 0.85 and back["top_k"] == 40
          and back["reasoning"] == "medium")
    check("and a file with a key is private",
          oct(os.stat(target).st_mode & 0o777) == "0o600",
          oct(os.stat(target).st_mode & 0o777))

    with open(target, "w", encoding="utf-8") as handle:
        handle.write('{"llm_host": "http://kept", "top_k": "twenty", "bogus": 1')
    broken = load(target)
    check("a truncated file falls all the way back", broken == DEFAULTS)

    with open(target, "w", encoding="utf-8") as handle:
        json.dump({"llm_host": "http://kept", "top_k": "twenty"}, handle)
    mixed = load(target)
    check("a value of the wrong type is dropped, the rest kept",
          mixed["llm_host"] == "http://kept" and mixed["top_k"] == 20,
          "top_k %r" % (mixed["top_k"],))

    check("the default path is OUTSIDE the checkout",
          os.path.dirname(path()) == os.path.dirname(app_dir())
          and not path().startswith(app_dir() + os.sep),
          path())

    os.environ["STRIKEPOSE_SETTINGS"] = os.path.join(folder, "elsewhere.json")
    check("and the environment can move it", path().endswith("elsewhere.json"))
    del os.environ["STRIKEPOSE_SETTINGS"]

    check("an unwritable directory is not fatal",
          save(values, "/proc/nope/settings.json") is None)

    shutil.rmtree(folder, ignore_errors=True)
    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
