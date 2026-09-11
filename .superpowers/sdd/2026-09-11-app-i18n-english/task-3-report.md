# Task 3 Report: Locale state and immediate Settings switching

Status: complete

## Changes

- Added `AppLocale` (`ZhCn`/`En`) with stable locale tags and made it the default `UiState` field.
- Added `catalog::install` with app-catalog parsing and Chinese framework fallback diagnostics.
- Wired the locale installer into startup before `AppState` creation.
- Added bilingual language settings copy and immediate Settings language buttons.
- Added locale and English catalog tests.

`set_locale` updates only `UiState.locale`, installs the selected catalog, and refreshes windows; it does not reset other state entities.

## Verification

- `cargo fmt --all -- --check` — passed.
- `cargo test --test catalog --test ui` — passed (18 catalog tests, 4 UI tests).
- `cargo check` — passed.

## Commit

The implementation is committed as `feat: add runtime language switcher`.

## Reviewer follow-up

- Changed the Chinese locale's English-language label from `English` to natural Chinese `英文`.
- Added exact assertions for all four language settings strings in both locale maps.
