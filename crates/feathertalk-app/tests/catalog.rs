use feathertalk_app::assets::Step;
use feathertalk_app::catalog;
use feathertalk_app::navigation::Page;
use feathertalk_app::tasks::{
    kind_key, recovery_key, stage_key, status_key, CRASHED_KEY, REJECTED_KEY, UNAVAILABLE_KEY,
    UNSUPPORTED_KEY,
};
use feathertalk_app::training::{
    mode_hint_key, mode_label_key, variant_label_key, ALL_MODES, ALL_VARIANTS,
};
use feathertalk_domain::{Recovery, TaskKind, TaskStage, TaskStatus};

/// `Recovery` has no `ALL`, so the list is spelled out here; a new variant makes
/// this test fail rather than leaking a key onto the screen.
const RECOVERIES: [Recovery; 7] = [
    Recovery::Retry,
    Recovery::ResumeFromCheckpoint,
    Recovery::FreeDiskSpace,
    Recovery::SelectDifferentAdapter,
    Recovery::ExcludeBadFrames,
    Recovery::ReimportModel,
    Recovery::NotRecoverable,
];

#[test]
fn english_and_unknown_locale_selection_are_distinct() {
    let english = catalog::translations("en").expect("the English catalog is valid");
    assert_eq!(english.get("shell.title"), Some("FeatherTalk Workbench"));
    assert!(matches!(
        catalog::translations("unknown"),
        Err(catalog::CatalogError::UnsupportedLocale(_))
    ));
}

#[test]
fn english_catalog_contains_language_settings_copy() {
    let english = catalog::translations("en").expect("the English catalog is valid");
    assert_eq!(english.get("ui.language.title"), Some("Language"));
    assert_eq!(
        english.get("ui.language.description"),
        Some("Choose the interface language.")
    );
    assert_eq!(english.get("ui.language.zh_cn"), Some("中文"));
    assert_eq!(english.get("ui.language.en"), Some("English"));
}

#[test]
fn chinese_catalog_contains_language_settings_copy() {
    let chinese = catalog::translations("zh-CN").expect("the Chinese catalog is valid");
    assert_eq!(chinese.get("ui.language.title"), Some("语言"));
    assert_eq!(
        chinese.get("ui.language.description"),
        Some("选择界面语言。")
    );
    assert_eq!(chinese.get("ui.language.zh_cn"), Some("中文"));
    assert_eq!(chinese.get("ui.language.en"), Some("英文"));
}

#[test]
fn the_bundled_catalog_parses() {
    let translations =
        catalog::translations("zh-CN").expect("the bundled catalog is a JSON object");
    assert!(matches!(
        catalog::translations("unknown"),
        Err(catalog::CatalogError::UnsupportedLocale(_))
    ));
    assert_eq!(catalog::LOCALE_TAG, "zh-CN");
    assert_eq!(translations.get("shell.title"), Some("FeatherTalk 工作台"));
}

#[test]
fn every_navigation_key_resolves_to_chinese_copy() {
    let translations =
        catalog::translations("zh-CN").expect("the bundled catalog is a JSON object");
    for page in Page::ALL {
        for key in [page.label_key(), page.title_key(), page.pending_key()] {
            let value = translations.get(key).unwrap_or_default();
            assert!(!value.is_empty(), "{key} has no copy");
            assert!(
                value.chars().any(|character| character >= '\u{4e00}'),
                "{key} is not Chinese copy: {value}"
            );
        }
    }
}

#[test]
fn every_shell_key_resolves() {
    let translations =
        catalog::translations("zh-CN").expect("the bundled catalog is a JSON object");
    for key in catalog::SHELL_KEYS {
        assert!(
            !translations.get(key).unwrap_or_default().is_empty(),
            "{key} has no copy"
        );
    }
}

#[test]
fn every_generate_page_key_resolves() {
    let translations =
        catalog::translations("zh-CN").expect("the bundled catalog is a JSON object");
    for key in catalog::GENERATE_PAGE_KEYS {
        assert!(
            !translations.get(key).unwrap_or_default().is_empty(),
            "{key} has no copy"
        );
    }
}

#[test]
fn the_catalog_holds_objects_and_non_empty_strings_only() {
    let value: serde_json::Value =
        serde_json::from_str(catalog::RAW).expect("the bundled catalog is valid JSON");
    let mut stack = vec![&value];
    while let Some(current) = stack.pop() {
        match current {
            serde_json::Value::Object(map) => stack.extend(map.values()),
            serde_json::Value::String(text) => {
                assert!(!text.trim().is_empty(), "the catalog holds an empty string")
            }
            other => panic!("the catalog holds objects and strings only: {other}"),
        }
    }
}

#[test]
fn every_protocol_enum_key_resolves_to_chinese_copy() {
    let translations =
        catalog::translations("zh-CN").expect("the bundled catalog is a JSON object");
    let mut keys = Vec::new();
    for status in TaskStatus::ALL {
        keys.push(status_key(status));
    }
    for stage in TaskStage::ALL_UNIT_SAMPLES {
        keys.push(stage_key(&stage));
    }
    for kind in TaskKind::ALL {
        keys.push(kind_key(kind));
    }
    for recovery in RECOVERIES {
        keys.push(recovery_key(recovery));
    }
    for key in keys {
        let value = translations.get(key).unwrap_or_default();
        assert!(!value.is_empty(), "{key} has no copy");
        assert!(
            value.chars().any(|character| character >= '\u{4e00}'),
            "{key} is not Chinese copy: {value}"
        );
    }
}

#[test]
fn every_task_page_key_resolves() {
    let translations =
        catalog::translations("zh-CN").expect("the bundled catalog is a JSON object");
    for key in catalog::TASKS_PAGE_KEYS {
        assert!(
            !translations.get(key).unwrap_or_default().is_empty(),
            "{key} has no copy"
        );
    }
}

#[test]
fn the_shells_own_failure_summaries_are_task_page_keys() {
    // The two keys `tasks.rs` spells out have to be in the checked table, or a
    // typo in either one shows up on screen instead of in a test.
    assert!(catalog::TASKS_PAGE_KEYS.contains(&CRASHED_KEY));
    assert!(catalog::TASKS_PAGE_KEYS.contains(&UNAVAILABLE_KEY));
    assert!(catalog::TASKS_PAGE_KEYS.contains(&REJECTED_KEY));
    assert!(catalog::TASKS_PAGE_KEYS.contains(&UNSUPPORTED_KEY));
}

#[test]
fn every_asset_page_key_resolves() {
    let translations =
        catalog::translations("zh-CN").expect("the bundled catalog is a JSON object");
    for key in catalog::ASSETS_PAGE_KEYS {
        assert!(
            !translations.get(key).unwrap_or_default().is_empty(),
            "{key} has no copy"
        );
    }
}

#[test]
fn every_asset_step_has_a_name_and_a_sentence() {
    // Walking `Step::ALL` rather than listing the keys again: a fifth step fails
    // this test instead of rendering a raw key.
    let translations =
        catalog::translations("zh-CN").expect("the bundled catalog is a JSON object");
    for step in Step::ALL {
        for key in [step.label_key(), step.hint_key()] {
            let value = translations.get(key).unwrap_or_default();
            assert!(!value.is_empty(), "{key} has no copy");
            assert!(
                value.chars().any(|character| character >= '\u{4e00}'),
                "{key} is not Chinese copy: {value}"
            );
        }
    }
}

#[test]
fn every_training_page_key_resolves() {
    let translations =
        catalog::translations("zh-CN").expect("the bundled catalog is a JSON object");
    for key in catalog::TRAINING_PAGE_KEYS {
        assert!(
            !translations.get(key).unwrap_or_default().is_empty(),
            "{key} has no copy"
        );
    }
}

#[test]
fn every_mode_and_variant_is_named() {
    // The three presets and the two variants come from the protocol's enums, so a
    // fourth mode upstream fails here instead of waiting for somebody to remember
    // a second list.
    let translations =
        catalog::translations("zh-CN").expect("the bundled catalog is a JSON object");
    let mut keys = Vec::new();
    for mode in ALL_MODES {
        keys.push(mode_label_key(mode));
        keys.push(mode_hint_key(mode));
    }
    for variant in ALL_VARIANTS {
        keys.push(variant_label_key(variant));
    }
    for key in keys {
        let value = translations.get(key).unwrap_or_default();
        assert!(!value.is_empty(), "{key} has no copy");
        assert!(
            value.chars().any(|character| character >= '\u{4e00}'),
            "{key} is not Chinese copy: {value}"
        );
    }
}

#[test]
fn every_locale_key_resolves() {
    for locale in catalog::LOCALE_TAGS {
        let translations = catalog::translations(locale).expect("locale catalog is valid");
        let mut keys = Vec::new();
        keys.extend(catalog::SHELL_KEYS);
        keys.extend(catalog::TASKS_PAGE_KEYS);
        keys.extend(catalog::ASSETS_PAGE_KEYS);
        keys.extend(catalog::TRAINING_PAGE_KEYS);
        keys.extend(catalog::GENERATE_PAGE_KEYS);
        for page in Page::ALL {
            keys.extend([page.label_key(), page.title_key(), page.pending_key()]);
        }
        for step in Step::ALL {
            keys.extend([step.label_key(), step.hint_key()]);
        }
        for status in TaskStatus::ALL {
            keys.push(status_key(status));
        }
        for stage in TaskStage::ALL_UNIT_SAMPLES {
            keys.push(stage_key(&stage));
        }
        for kind in TaskKind::ALL {
            keys.push(kind_key(kind));
        }
        for recovery in RECOVERIES {
            keys.push(recovery_key(recovery));
        }
        for mode in ALL_MODES {
            keys.extend([mode_label_key(mode), mode_hint_key(mode)]);
        }
        for variant in ALL_VARIANTS {
            keys.push(variant_label_key(variant));
        }
        for key in keys {
            assert!(
                !translations.get(key).unwrap_or_default().trim().is_empty(),
                "{locale}: {key} has no copy"
            );
        }
    }
}

#[test]
fn english_catalog_has_no_placeholder_copy() {
    let forbidden = [
        "Section",
        "Additional information",
        "Additional guidance",
        "Select an option",
        "Cancel hint",
        "Epochs hint",
        "Fast hint",
        "Mouth hint",
        "Temporal hint",
        "Resume hint",
        "Package title",
        "Package description",
        "Monitoring title",
        "Monitoring description",
        "Metrics empty",
        "Metrics details hint",
        "Preview title",
        "Preview description",
        "Full title",
        "Full description",
        "Format description",
        "Parent hint",
        "File hint",
        "ONNX hint",
        "CPU hint",
        "GPU hint",
        "Refresh hint",
        "No match hint",
        "Force cancel hint",
    ];
    for raw in [
        include_str!("../locales/en/base.json"),
        include_str!("../locales/en/ui.json"),
        include_str!("../locales/en/workflow.json"),
        include_str!("../locales/en/models.json"),
    ] {
        let value: serde_json::Value = serde_json::from_str(raw).expect("English JSON is valid");
        let mut stack = vec![value];
        while let Some(current) = stack.pop() {
            match current {
                serde_json::Value::Object(map) => stack.extend(map.into_values()),
                serde_json::Value::String(text) => {
                    assert!(
                        !forbidden.contains(&text.as_str()),
                        "forbidden placeholder: {text}"
                    );
                }
                _ => {}
            }
        }
    }
}

#[test]
fn english_catalog_uses_distinct_page_and_model_labels() {
    let base: serde_json::Value =
        serde_json::from_str(include_str!("../locales/en/base.json")).expect("base JSON");
    let models: serde_json::Value =
        serde_json::from_str(include_str!("../locales/en/models.json")).expect("models JSON");
    let page_titles = [
        ("assets", "Asset Preparation"),
        ("training", "Model Training"),
        ("generate", "Video Generation"),
        ("models", "Model Tools"),
        ("tasks", "Task History"),
    ];
    for (page, expected) in page_titles {
        assert_eq!(base["page"][page]["title"], expected, "base page title");
    }
    let contextual_labels = [
        ("assets", "project", "Project directory"),
        ("assets", "video", "Source video"),
        ("training", "device", "Compute device"),
        ("training", "output", "Output directory"),
    ];
    for (section, field, expected) in contextual_labels {
        assert_eq!(
            base[section][field]["label"], expected,
            "contextual field label"
        );
    }
    let operations = [
        ("inspect", "Inspect Model"),
        ("import", "Import Legacy Model"),
        ("package", "Export Model Package"),
        ("onnx", "Export ONNX"),
        ("features", "Migrate Legacy Features"),
    ];
    for (operation, expected) in operations {
        assert_eq!(
            models[format!("models.operation.{operation}")],
            expected,
            "model operation label"
        );
    }
}

#[test]
fn english_catalog_has_contextual_paths_and_no_generic_placeholders() {
    let models: serde_json::Value =
        serde_json::from_str(include_str!("../locales/en/models.json")).expect("models JSON");
    assert_eq!(
        models["models.source.legacy"],
        "Legacy weights file (.pth or .pth.tar)"
    );
    assert_eq!(
        models["models.source.checkpoint"],
        "Training checkpoint directory"
    );
    assert_eq!(models["models.destination.onnx"], "ONNX output file path");
    assert_eq!(
        models["models.destination.features"],
        "Versioned features output file path"
    );

    let forbidden = [
        "Overview",
        "See details for this option.",
        "Batch size hint",
        "Section",
        "Additional information",
        "Additional guidance",
        "Select an option",
        "Information",
        "See the guidance above.",
        "No items available",
    ];
    for raw in [
        include_str!("../locales/en/base.json"),
        include_str!("../locales/en/ui.json"),
        include_str!("../locales/en/workflow.json"),
        include_str!("../locales/en/models.json"),
    ] {
        let value: serde_json::Value = serde_json::from_str(raw).expect("English JSON is valid");
        let mut stack = vec![value];
        while let Some(current) = stack.pop() {
            match current {
                serde_json::Value::Object(map) => stack.extend(map.into_values()),
                serde_json::Value::String(text) => {
                    assert!(
                        !forbidden.contains(&text.as_str()),
                        "generic placeholder: {text}"
                    );
                }
                _ => {}
            }
        }
    }
}

#[test]
fn english_catalog_has_actionable_model_errors_and_task_gates() {
    let base: serde_json::Value =
        serde_json::from_str(include_str!("../locales/en/base.json")).expect("base JSON");
    let models: serde_json::Value =
        serde_json::from_str(include_str!("../locales/en/models.json")).expect("models JSON");

    let model_errors = [
        (
            "models.error.no_source",
            "Choose a source file or model directory before submitting this operation.",
        ),
        (
            "models.error.no_destination",
            "Choose an output location before submitting this operation.",
        ),
        (
            "models.error.same_path",
            "Choose a different output location; it cannot be the same as the source path.",
        ),
        (
            "models.error.destination_absolute",
            "Please choose the complete output path again.",
        ),
        (
            "models.error.destination_inside_source",
            "Save the output outside the source model directory to preserve the source model's complete directory structure.",
        ),
        (
            "models.error.destination_name",
            "Choose a specific new directory or file; the disk root cannot be used.",
        ),
        (
            "models.error.legacy_extension",
            "Choose a legacy weights file with a .pth or .pth.tar extension.",
        ),
        (
            "models.error.features_extension",
            "Choose a legacy features file with a lowercase .npy extension.",
        ),
        (
            "models.error.destination_exists",
            "The output location already exists. Choose a different output location and try again.",
        ),
        (
            "models.error.destination_unavailable",
            "The output location cannot be accessed. Check the path and permissions.",
        ),
        (
            "models.error.destination_parent",
            "The output location's parent must be an existing directory. Choose another location.",
        ),
        (
            "models.error.picker",
            "The file picker could not complete. Please try again.",
        ),
        (
            "models.error.details",
            "Review the error details below for more information.",
        ),
    ];
    for (key, expected) in model_errors {
        assert_eq!(models[key], expected, "actionable model error copy");
    }

    let task_gates = [
        (
            "no_project",
            "Choose a project directory on the Assets page before submitting a task.",
        ),
        (
            "no_worker",
            "The local compute service is unavailable. Open Settings > Local compute service to configure it.",
        ),
        (
            "busy",
            "A task is already running; only one task can run at a time.",
        ),
    ];
    for (key, expected) in task_gates {
        assert_eq!(base["tasks"]["blocked"][key], expected, "task gate copy");
    }

    let forbidden_model_values = [
        "Destination absolute",
        "Destination inside source",
        "Destination name",
        "Legacy extension",
        "Features extension",
        "Destination exists",
        "Destination unavailable",
        "Destination parent",
        "Picker",
        "Details",
    ];
    for value in model_errors.iter().map(|(key, _)| &models[*key]) {
        assert!(
            !forbidden_model_values
                .iter()
                .any(|forbidden| value == forbidden),
            "terse model error copy is forbidden: {value}"
        );
    }

    let forbidden_task_gate_values = ["No project", "No worker", "Busy"];
    for value in task_gates
        .iter()
        .map(|(key, _)| &base["tasks"]["blocked"][*key])
    {
        assert!(
            !forbidden_task_gate_values
                .iter()
                .any(|forbidden| value == forbidden),
            "terse task gate copy is forbidden: {value}"
        );
    }
}
