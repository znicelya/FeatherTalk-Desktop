# FeatherTalk Desktop English i18n Design

## Goal

Add complete English UI support to `crates/feathertalk-app` while preserving
Chinese as the default language. Users can switch between Chinese and English
from the existing settings page, and the current page updates immediately.

## Resource layout

Organize application translations by locale:

```text
crates/feathertalk-app/locales/
├── zh-CN/
│   ├── base.json
│   ├── ui.json
│   ├── workflow.json
│   └── models.json
└── en/
    ├── base.json
    ├── ui.json
    ├── workflow.json
    └── models.json
```

Each locale contains the same key set. The existing Chinese files are moved
without changing their translated values; English files provide natural,
consistent copy for every application key.

## Runtime architecture

- `catalog` exposes locale tags and builds a `TranslationMap` for a requested
  locale by parsing its four embedded JSON files.
- Startup installs the Chinese catalog (`zh-CN`) as today.
- `UiState` stores the active application locale for rendering the selected
  language in settings.
- A shared locale-switch helper installs the selected catalog with
  `locale::install_with_translations` and calls `cx.refresh_windows()`.
- Switching language does not recreate or reset project, task, form, compute,
  theme, or navigation state.
- Yororen UI framework translations remain available through its bundled
  locale catalogs; application translations layer on top.

## Settings UI

Add a language section to the settings page with two choices:

- 中文 (`zh-CN`)
- English (`en`)

The selected choice uses the existing primary/neutral button styling. The
section title and descriptions are translated in both catalogs. Changing the
selection updates all visible text immediately, including the window title.

## Validation

- Parse both locale catalogs during tests.
- Verify every key listed by the existing catalog key tables resolves to a
  non-empty value in both locales.
- Verify enum-derived page, task, training, and model keys resolve in both
  locales.
- Run the app crate's complete `cargo test` and `cargo check`.

## Scope boundaries

- Default language remains Chinese.
- No language persistence or OS-locale auto-detection is added in this change.
- No changes are made to worker diagnostics written to stderr; those remain
  English technical output as they are today.
