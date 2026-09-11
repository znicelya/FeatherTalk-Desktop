# FeatherTalk Desktop English i18n Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete English application catalog and an immediate Chinese/English switch in Settings while keeping Chinese as the default.

**Architecture:** Locale resources are grouped under `locales/zh-CN/` and `locales/en/`, each with base, UI, workflow, and model JSON files. The catalog module parses a requested locale, and a single switch helper installs that catalog and refreshes all windows. `UiState` owns the active locale so Settings can render the selected option without touching task or project state.

**Tech Stack:** Rust 2024, gpui-ce 0.3.3, yororen_ui 0.3.0 i18n runtime, embedded JSON catalogs, Cargo tests.

## Global Constraints

- Default language remains `zh-CN`.
- No OS-locale auto-detection or persistence is added.
- Both locales must expose the same application key set.
- Framework translations continue to come from Yororen UI bundled catalogs.
- Worker stderr diagnostics remain unchanged.

---

### Task 1: Reorganize locale resources and make catalog locale-aware

**Files:**
- Create: `crates/feathertalk-app/locales/zh-CN/base.json` (move existing `zh-CN.json` contents)
- Create: `crates/feathertalk-app/locales/zh-CN/ui.json` (move existing `ui.zh-CN.json` contents)
- Create: `crates/feathertalk-app/locales/zh-CN/workflow.json` (move existing `workflow.zh-CN.json` contents)
- Create: `crates/feathertalk-app/locales/zh-CN/models.json` (move existing `models.zh-CN.json` contents)
- Modify: `crates/feathertalk-app/src/catalog.rs`
- Test: `crates/feathertalk-app/tests/catalog.rs`

**Interfaces:**
- Produce `pub const LOCALE_TAGS: &[&str] = &["zh-CN", "en"];`.
- Produce `pub fn translations(locale_tag: &str) -> Result<TranslationMap, CatalogError>`.
- Preserve `catalog::translations("zh-CN")` as the Chinese parser used by startup and tests.

- [ ] **Step 1: Add a red catalog API test**

Add a test that calls `catalog::translations("zh-CN")`, asserts `shell.title` is present, and asserts an unknown locale returns `CatalogError` rather than silently using Chinese.

- [ ] **Step 2: Run the focused catalog test and verify it fails**

Run `cargo test --test catalog the_bundled_catalog_parses` from `crates/feathertalk-app`.
Expected: compile failure because `translations` currently takes no locale argument.

- [ ] **Step 3: Move the four Chinese files and update `catalog.rs`**

Use `Move-Item` only for the four exact files, then update `RAW`/`ADDITIONAL` into locale-specific constants and parse the selected set. Keep `SHELL_KEYS`, page key tables, and `CatalogError` unchanged except for locale validation. Return a new `CatalogError::UnsupportedLocale(String)` for tags other than `zh-CN` and `en`.

- [ ] **Step 4: Update existing call sites to pass `"zh-CN"`**

Change startup and layout tests to call `catalog::translations(catalog::LOCALE_TAG)` (or the explicit Chinese tag where the test is intentionally Chinese-only).

- [ ] **Step 5: Run the focused test and commit**

Run `cargo test --test catalog the_bundled_catalog_parses`.
Expected: PASS. Commit with `refactor: organize locale catalogs by language`.

### Task 2: Add the complete English catalogs

**Files:**
- Create: `crates/feathertalk-app/locales/en/base.json`
- Create: `crates/feathertalk-app/locales/en/ui.json`
- Create: `crates/feathertalk-app/locales/en/workflow.json`
- Create: `crates/feathertalk-app/locales/en/models.json`
- Modify: `crates/feathertalk-app/src/catalog.rs`
- Test: `crates/feathertalk-app/tests/catalog.rs`

**Interfaces:**
- `catalog::translations("en")` returns a map containing every key exposed by the existing key tables plus all enum-derived keys.
- `catalog::translations("zh-CN")` continues returning the moved Chinese catalog unchanged.

- [ ] **Step 1: Add a red parity test**

Add a helper test that loops over `catalog::LOCALE_TAGS`, parses each map, and asserts every key in `SHELL_KEYS`, `TASKS_PAGE_KEYS`, `ASSETS_PAGE_KEYS`, `TRAINING_PAGE_KEYS`, and `GENERATE_PAGE_KEYS` is non-empty. Also walk `Page::ALL`, `Step::ALL`, `TaskStatus::ALL`, `TaskStage::ALL_UNIT_SAMPLES`, `TaskKind::ALL`, `ALL_MODES`, `ALL_VARIANTS`, and the recovery list exactly as existing tests do.

- [ ] **Step 2: Run the parity test and verify it fails for English**

Run `cargo test --test catalog every_locale_key_resolves`.
Expected: failure because the English files do not exist yet.

- [ ] **Step 3: Write English JSON resources**

Translate every existing Chinese resource value into concise English while preserving object shape, placeholders, technical names (`ONNX`, `FeatherHuBERT`, `worker`, `epoch`, `step`), and punctuation semantics. Keep all four files valid JSON objects containing strings only.

- [ ] **Step 4: Extend catalog parsing for English fragments**

Point the locale selector in `catalog.rs` at `locales/en/base.json`, `en/ui.json`, `en/workflow.json`, and `en/models.json`.

- [ ] **Step 5: Run parity and JSON tests, then commit**

Run `cargo test --test catalog`.
Expected: all catalog tests pass for both locales. Commit with `feat: add English application translations`.

### Task 3: Add locale state and immediate Settings switching

**Files:**
- Modify: `crates/feathertalk-app/src/ui.rs`
- Modify: `crates/feathertalk-app/src/catalog.rs`
- Modify: `crates/feathertalk-app/src/main.rs`
- Modify: `crates/feathertalk-app/src/components/settings.rs`
- Modify: `crates/feathertalk-app/locales/zh-CN/ui.json`
- Modify: `crates/feathertalk-app/locales/en/ui.json`
- Test: `crates/feathertalk-app/tests/catalog.rs`

**Interfaces:**
- `UiState` gains `pub locale: AppLocale`, defaulting to `AppLocale::ZhCn`.
- `AppLocale` is a `Copy + Eq` enum with `ZhCn` and `En`, plus `tag() -> &'static str`.
- `catalog::install(cx, locale: AppLocale)` installs the selected framework locale and merges the selected app map, logging and falling back to Chinese only on catalog parse failure.

- [ ] **Step 1: Add state and switching tests**

Add unit tests for `AppLocale::default().tag() == "zh-CN"`, both enum tags, and a catalog test asserting the English map contains the new settings language keys (`ui.language.title`, `ui.language.description`, `ui.language.zh_cn`, `ui.language.en`).

- [ ] **Step 2: Implement `AppLocale` and catalog installation**

Define the enum in `ui.rs`. In `catalog.rs`, add `install(cx: &mut App, locale: AppLocale)` that calls `locale::install_with_translations(cx, locale.tag(), translations(locale.tag())?)`; on error, print the existing diagnostic and install Chinese. Keep `LOCALE_TAG` as the Chinese default alias for compatibility.

- [ ] **Step 3: Wire startup through the new installer**

Replace `install_catalog(cx)` with `catalog::install(cx, AppLocale::default())` in `main.rs`; retain the current startup order before creating `AppState`.

- [ ] **Step 4: Add translated language strings**

Add the four language settings keys to both `ui.json` files. The English values must be `Language`, `Choose the interface language.`, `中文`, and `English`; Chinese values must be natural Chinese equivalents.

- [ ] **Step 5: Render and handle language controls in Settings**

Add a `settings-language` section with two buttons. Read `state.ui.read(cx).locale`, style the selected button as primary, and on click call a shared `set_locale(locale, cx)` helper. The helper updates `UiState.locale`, installs the catalog, and calls `cx.refresh_windows()`; it must not alter other state entities.

- [ ] **Step 6: Run UI-adjacent tests and commit**

Run `cargo test --test catalog --test ui` and `cargo check` from `crates/feathertalk-app`.
Expected: PASS. Commit with `feat: add runtime language switcher`.

### Task 4: Full verification and regression coverage

**Files:**
- Modify: `crates/feathertalk-app/tests/catalog.rs` (only if parity assertions need final cleanup)

- [ ] **Step 1: Run formatting and diff checks**

Run `cargo fmt --check` and `git diff --check` from the app crate/repository root. Fix only formatting introduced by this feature.

- [ ] **Step 2: Run the complete app test suite**

Run `cargo test` from `crates/feathertalk-app`.
Expected: every test target passes with zero failures.

- [ ] **Step 3: Run the final compile check**

Run `cargo check` from `crates/feathertalk-app`.
Expected: exit code 0.

- [ ] **Step 4: Inspect the final diff and commit any test-only cleanup**

Run `git status --short` and `git diff --stat`; verify only locale resources, catalog/runtime wiring, settings UI, tests, and the plan/spec docs are changed. If cleanup is needed, commit it with `test: verify bilingual locale coverage`.
