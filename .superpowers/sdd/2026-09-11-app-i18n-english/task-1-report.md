# Task 1 report

## Status

Implemented and committed.

## Files changed

- Reorganized the four Chinese locale resources under `crates/feathertalk-app/locales/zh-CN/` as `base.json`, `ui.json`, `workflow.json`, and `models.json`.
- Updated `src/catalog.rs` with `LOCALE_TAGS`, locale-parameterized `translations`, and `CatalogError::UnsupportedLocale`.
- Updated startup and layout test call sites to pass the Chinese locale tag.
- Added catalog coverage asserting the Chinese catalog parses and unsupported locales return an error.

## Verification

- `cargo test --test catalog the_bundled_catalog_parses` — PASS (1 passed, 0 failed)
- `cargo test --test catalog` — PASS (12 passed, 0 failed)

## Commit

Commit message: `refactor: organize locale catalogs by language`

## Concerns

English resource files are intentionally supplied by the subsequent English-catalog task. Until those files are added, `translations("en")` remains rejected as unsupported; this task preserves the existing Chinese behavior while establishing the locale-aware API.

## Reviewer fix

Added locale-specific English resource selection and minimal valid English placeholders. `translations("en")` now parses successfully, while unknown tags return `UnsupportedLocale`.

- `cargo fmt --all` — PASS
- `cargo test --test catalog` — PASS (13 passed, 0 failed)
