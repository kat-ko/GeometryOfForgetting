# third_party/patches

`.patch` files for the **only** case where a correction cannot live in
`src/glue/adapters/` and an in-place edit of vendored source is unavoidable.

Rules:
- Prefer the adapter layer (`src/glue/adapters/`). A patch here is a last resort.
- One `.patch` per vendored repo, named `{repo}.patch`, applied on top of the
  pinned SHA recorded in `../VENDORED.md`.
- Reference every patch from `../VENDORED.md` (repo section) with a one-line reason.
- The estimator version string becomes `{repo}@{sha}+{patch-name}` once a patch is
  applied, so patched and unpatched numbers never pool (`docs/02-validation-suite.md` §9).

Empty by design. If this directory has only this README, no vendored source has
been modified — which is the intended state.
