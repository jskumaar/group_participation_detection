# Gaze pipeline onboarding site

A static site for new developers on this project. There is no build step:
`index.html` and `assets/` are served as they are by GitHub Pages from the
`gh-pages` branch.

## Update it after the code changes

The site is pinned to one commit, so its links and snippets never drift. To move
it to a newer commit, run this from a checkout of `gh-pages`, with a path to a
checkout of the code:

```bash
python tools/gen_refs.py <path-to-repo> main
```

That rewrites `assets/js/refs.js`. It stops with an error if a function it links
to has been renamed or moved, so fix the pattern in `tools/gen_refs.py`. Then
re-read any page text about the code that changed.

## Keep the demos honest

`assets/js/model.js` holds JavaScript ports of the project's maths, and every demo
calls it. Two checks compare it with the real code. Run them if you change that
maths in either place.

Against the C++ (needs CMake, Visual Studio and OpenCV):

```powershell
cmake -S tools/port_check -B build_port -G "Visual Studio 17 2022" -A x64 "-DC_IMPL=<path-to-repo>/src/c_impl"
cmake --build build_port --config Release
node tools/port_check/compare.js gen cases.txt
build_port\Release\port_check.exe cases.txt > out.txt
node tools/port_check/compare.js check cases.txt out.txt
```

Against the Python client:

```bash
python tools/check_scoring.py <path-to-repo>/src/python_client
```

## Rules for this site

- This site is public. Use synthetic scenes only: no study footage, participant
  data or real labels.
- Blue, orange and aqua always mean P1, P2 and P3, so do not use them for anything
  else. Text labels go with them, so colour is never the only signal.
